# coding: utf-8

import logging
import os
import shutil
from datetime import date
from datetime import datetime
from typing import List, Dict
from typing import Union

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

from cfg.paths import DIR_CFG
from cfg.paths import DIR_DATA
from cfg.paths import DIR_MAPPINGS, DIR_CACHE
from src.data_prep import get_latest_di_tradeflow
from src.data_wrangling import SELECTED_SKUS_DC
from src.scenario import *
from src.utils.impala_utils import ImpalaUtils
from src.utils.sql_utils import SQLUtils

V_QUALITATIVE = '0'

V_QUANTITATIVE = '1'

F_QUALITY_CHECKS = 'QUALITY_CHECKS'

F_BASENAME = 'basename'
F_OUTPUT = 'output'
F_TRANSFORMED_PATH = 'transformed_path'
F_COPY_PATH = "copy_path"
F_SOURCE_PATH = "source_path"
F_FILEPATH = 'filepath'
F_FILENAME = 'filename'
F_FILE_ID = 'file_id'
F_FILE_TIMESTAMP = 'file_timestamp'

logger = logging.getLogger(__name__)

C_CYCLE_MONTH_LAG = 1  # 0 means cycle month is this calendar month; 1 means cycle month is next calendar month
C_IL_RANGE = 2  # Farthest month whose actual is updated for IL (e.g. 2 means actuals starting from M-2 are updated)
# note: when changing both numbers, make sure input files contain data of corresponding months

if C_CYCLE_MONTH_LAG not in [-1, 0, 1]:
    raise ValueError('Cycle month not recognized. Please specify the correct paramter for cycle.')


class DataPrep:

    # Constructors
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.eib_sellin = None
        self.di_tradeflow = None
        self.il_offtake = None
        self.tst = datetime.today().strftime(F_DN_DATE_FMT)

    def prep_smartpath(self, file_id: str = 'smartpath') -> None:
        """
        The purpose of this function is to load in the non-tabular smartpath competitor data, transform the data and
        output the data as an input of the model
        :param file_id: this is the corresponding key of the dictionary specified in data_prep.yml, which contains the
        path information, file name information and a user-defined hierarchical structure template used to load the
        data
        :return: None; the dataframe generated will be output to a csv. file on smartdata directly
        """

        # load the smartpath competitor raw data in non-tabular form
        smartpath_df = self.load_latest_file(file=file_id)

        # Split the dataframe using the user-defined hierarchical structure template
        # Read in the hierarchical template dictionary in data_prep.yml
        smartpath_hierarchy_dict = self.cfg[file_id]['hierarchy']

        # All the keys specified in the template can be found under column 'Import analysis'
        smartpath_df_pseudo_index = smartpath_df.loc[:, 'Import analysis'].apply(lambda x: str(x).strip())
        smartpath_df = smartpath_df.loc[:, self.cfg[file_id]['timeaxis']['columns']['start']: self.cfg[file_id][
            'timeaxis']['columns']['end']]
        smartpath_df.columns = smartpath_df.loc[
                               smartpath_df_pseudo_index == self.cfg[file_id]['timeaxis']['row'],
                               self.cfg[file_id]['timeaxis']['columns']['start']:
                               self.cfg[file_id]['timeaxis']['columns']['end']
                               ].iloc[0].tolist()

        smartpath_df['Import analysis'] = smartpath_df_pseudo_index

        def read_chunk(url_stack: list) -> pd.DataFrame:
            """
            The purpose of this function is to read in data in chunks (i.e., fill in the values for each key specified
            in the template dictionary
            :param url_stack: list of dictionary keys
            :return: a pandas dataframe stored as the values of the specified key in the specified template
            """
            chunk_df = pd.DataFrame()
            level, please_read = 0, False
            for _i, row in smartpath_df.iterrows():
                row_value_0 = row['Import analysis']

                if please_read is True:
                    if row_value_0 == 'nan':
                        return chunk_df
                    else:
                        chunk_df = chunk_df.append(row)

                else:
                    if row_value_0 == url_stack[level]:
                        level += 1

                    if level == len(url_stack):
                        please_read = True

            return chunk_df

        def rec_search_values(hierarchy_dict, url_stack: list) -> List[Dict]:
            """
            This is function is used to read in all the chunks of data as the values of the keys in the hierarchical
            dictionary
            :param hierarchy_dict: hierarchical dictionary
            :param url_stack: list of keys
            :return: a list of dictionaries
            """
            if len(hierarchy_dict) == 0:
                return [{
                    'url': url_stack,
                    'dataframe': read_chunk(url_stack=url_stack)
                }]

            df_list = []
            for key in hierarchy_dict:
                df_list.extend(
                    rec_search_values(hierarchy_dict=hierarchy_dict[key], url_stack=url_stack + [key])
                )
            return df_list

        dataframes: List[Dict] = rec_search_values(
            hierarchy_dict=smartpath_hierarchy_dict,
            url_stack=[]
        )

        # Transformation
        # step 1: remove comma in numeric columns
        for index in range(len(dataframes)):
            dataframes[index]['dataframe'] = self.remove_comma_in_numeric_columns(dataframes[index]['dataframe'],
                                                                                  'Import analysis')
        # step 2; generate SP_value and SP_volume data with initial cleaning/transformations
        # get a list for value data related dictionaries and volume data related dictionaries
        list_of_sp_value_dict = [d for d in dataframes if d['url'][0] == "Value (RMB '000)"]
        list_of_sp_volume_dict = [d for d in dataframes if d['url'][0] == "Volume (Ton)"]

        def initial_transformation(chunk_dict, value_name: str) -> pd.DataFrame:
            """
            The purpose of this function is to conduct some initial transformation of the data while reading in
            the dictionaries from the list of dictionaries
            :param chunk_dict: this is the dictionary containing the chunk of data needs to be read in
            :param value_name: indicates the val_col after unpivoting the data
            :return: pandas dataframe with volume/value information
            """
            # read in df
            chunk_df = chunk_dict['dataframe']
            # change column name for the brand info
            chunk_df = chunk_df.rename(columns={'Import analysis': 'Brand'})
            # columns to unpivot
            value_vars = list(chunk_df)
            value_vars.remove('Brand')
            # unpivot df
            df_long = pd.melt(chunk_df, id_vars='Brand', value_vars=value_vars, var_name='date', value_name=value_name)
            # add channel info
            df_long['channel'] = chunk_dict['url'][
                -1]  # the last level in the hierarchical structure indicates the channel
            df_long = df_long[df_long['Brand'].notnull()]
            # reformat date
            df_long['date'] = pd.to_datetime([datetime.strptime(date_val, '%b%Y') for date_val in df_long['date']])

            return df_long

        def generate_df(list_of_dict: list, value_name: str) -> pd.DataFrame:
            """
            The purpose of this function is to read in the dictionaries in the list of volume/value dictionaries
            defined above
            :param list_of_dict: list of sp value/volume dictionaries
            :param value_name: indicates the val_col after unpivoting the data
            :return: df after initial transformations
            """
            transformed_df = pd.DataFrame()
            for d in list_of_dict:  # dataframes is a list of dictionaries
                df_long = initial_transformation(d, value_name)
                transformed_df = pd.concat([transformed_df, df_long])

            return transformed_df

        # now read in all value/volume related raw data, and concatenate them to two dataframes
        sp_value = generate_df(list_of_sp_value_dict, "value_kRMB")
        sp_volume = generate_df(list_of_sp_volume_dict, 'volume_ton')

        # step 3: standardize brand scope, country and tier cols
        sp_value = self.sku_name_to_scope_brand_country_tier_source_cols(sp_value, 'Brand')
        sp_volume = self.sku_name_to_scope_brand_country_tier_source_cols(sp_volume, 'Brand')

        # step 4: select records with volume & value >=0
        sp_volume = sp_volume.query('volume_ton>=0')
        sp_value = sp_value.query('value_kRMB>=0')

        # step 5: merge value and volume info by defined merge_cols
        merge_cols = ['Brand', 'date', 'channel', 'Scope', 'Country', 'Tier', 'Source', 'Geography']
        sp = pd.merge(sp_value, sp_volume, on=merge_cols)

        # step 6: aggregate volume and value based on specified cols, and calculate price
        cols = ['Brand', 'date', 'channel', 'Scope', 'Country', 'Tier', 'Source', 'Geography']
        f = {'volume_ton': 'sum', 'value_kRMB': 'sum'}
        sp = sp.groupby(cols, as_index=False).agg(f)
        sp['price_kRMB_per_ton'] = sp['value_kRMB'] / sp['volume_ton']

        # step 7: reorder the cols
        sp['type'] = 'offtake'
        sp['volume_unit'] = 'ton'
        order_cols = ['Brand', 'type', 'Scope', 'Country', 'channel', 'date', 'volume_ton', 'volume_unit', 'value_kRMB',
                      'price_kRMB_per_ton', 'Source']
        sp = sp.loc[:, order_cols]

        # step 8: for all records with volume = 0 and value = 0, set price = 0
        df = sp.copy()
        df['price_kRMB_per_ton'].fillna(0, inplace=True)

        # step 9: select IL and selected channels
        _list_channels = ['B2C', 'BBC', 'C2C']
        df_il = df.query(
            '(Scope == "IL")&(Country == "Total")&(Source == "Smartpath")&(channel in @_list_channels)'
        ).copy()

        # step 10: rename columns:
        df_il.drop('Source', axis=1, inplace=True)
        df_il.rename(columns={'channel': 'Channel',
                              'volume_ton': 'volume',
                              'value_kRMB': 'value',
                              'price_kRMB_per_ton': 'price'}, inplace=True)

        # added by ZX, make sure we have right date format
        if str(os.name) == 'nt':
            df_il['date'] = pd.to_datetime(df_il['date']).dt.strftime('%#m/%#d/%Y')
        else:
            df_il['date'] = pd.to_datetime(df_il['date']).dt.strftime('%-m/%-d/%Y')

        # output file
        self.output_file(df_il, file_id)

    def prep_osa_eib(self, file_id: str = 'osa_eib'):
        """
        The purpose of this method is to prepare EIB OSA data
        """
        osa_eib_df = self.load_latest_file(file=file_id)

        # Find where the data table actually starts
        nan_rows = osa_eib_df.iloc[:, 0].isnull()

        first_data_row = nan_rows.idxmin()
        month_row = first_data_row - 2
        last_data_row = nan_rows.iloc[first_data_row:].idxmax()

        osa_eib_df = osa_eib_df.iloc[month_row:last_data_row]

        # Transform literal month name and yyyy to datetime
        osa_eib_df.iloc[0, :] = osa_eib_df.iloc[0, :].apply(
            lambda s: pd.to_datetime(s, format='%B %Y', errors='ignore'))
        osa_eib_df.iloc[0, :] = osa_eib_df.iloc[0, :].apply(
            lambda s: pd.to_datetime(s, format='%B %y', errors='ignore'))
        osa_eib_df.iloc[0, :] = osa_eib_df.iloc[0, :].apply(
            lambda s: pd.to_datetime(s, format='%b %Y', errors='ignore'))
        osa_eib_df.iloc[0, :] = osa_eib_df.iloc[0, :].apply(
            lambda s: pd.to_datetime(s, format='%b %y', errors='ignore'))
        osa_eib_df.iloc[0, :] = osa_eib_df.iloc[0, :].fillna(method='pad')
        osa_eib_df.iloc[0, :] = osa_eib_df.iloc[0, :].astype('datetime64[ns]').dt.strftime('%Y-%m')

        # Transform week format Wxx into int
        osa_eib_df.iloc[1, 1:] = osa_eib_df.iloc[1, 1:].str[1:].astype(int)

        # Change table format from wide to long
        osa_eib_df.rename({osa_eib_df.columns[0]: 'Items'}, axis=1, inplace=True)
        osa_eib_df = osa_eib_df.set_index('Items')
        osa_eib_df.columns = pd.MultiIndex.from_arrays([osa_eib_df.iloc[0, :], osa_eib_df.iloc[1, :]],
                                                       names=['month', 'week'])
        osa_eib_df = osa_eib_df.iloc[2:, :]

        osa_eib_df = osa_eib_df.T.stack()
        osa_eib_df.name = 'value'
        osa_eib_df = osa_eib_df.reset_index(['month', 'week']).loc[:, ['value', 'week', 'month']]

        osa_eib_df['Item'] = osa_eib_df.index

        # output file
        osa_eib_df = osa_eib_df[['Item', 'value', 'week', 'month']]
        self.output_file(osa_eib_df, file_id)

    def prep_price_eib(self, file_id: str = 'price_eib'):
        """
        The purpose of this method is to prepare EIB price
        """
        # Warning: one column in the data file contains a typo: 'Source counry'
        price_df = self.load_latest_file(file=file_id, header=None, skip_blank_lines=True)

        lookup_df = pd.read_csv(
            os.path.join(DIR_MAPPINGS, self.cfg[file_id]['brand_and_tier_mapping_sheet']),
            encoding='ISO-8859-1'
        )
        sku_mapping = pd.read_csv(
            os.path.join(DIR_MAPPINGS, self.cfg[file_id]['sku_mapping']),
            encoding='ISO-8859-1'
        )

        brand_mapping = lookup_df.loc[:, ['Brand', 'Brand_acc']].dropna(axis=0)
        brand_mapping = brand_mapping[brand_mapping['Brand'] != brand_mapping['Brand_acc']].set_index('Brand').squeeze()

        tier_mapping = lookup_df.loc[:, ['Tier', 'Tier_acc']].dropna(axis=0)
        tier_mapping = tier_mapping[tier_mapping['Tier'] != tier_mapping['Tier_acc']].set_index('Tier').squeeze()

        sku_mapping = sku_mapping[~sku_mapping.Stage_acc.isnull()].drop_duplicates(['SKU No', 'Tier'])

        price_df.rename(price_df.iloc[0, :], axis=1, inplace=True)
        price_df = price_df.iloc[1:, :]
        price_df = price_df[~price_df['SKU'].isnull()]

        price_df['Source counry'] = price_df['Source counry'].replace({'GE': 'DE'})
        price_df['Brand'] = price_df['Brand'].replace(brand_mapping)
        price_df['Tier'] = price_df['Tier'].replace(tier_mapping)

        price_df['SKU'] = pd.to_numeric(price_df['SKU'], errors='coerce')

        # Merge on SKU only
        price_df = price_df.merge(
            sku_mapping.loc[:, ['Country_acc', 'Brand_acc', 'Tier_acc', 'SKU No', 'Stage_acc', 'SKU_wo_pkg']],
            left_on=['SKU'], right_on=['SKU No'],
            how='left')

        # Drop rows with Tier conflicts
        price_df = price_df[(price_df.Tier == price_df.Tier_acc) | (price_df['Source counry'] == 'Source counry')]

        price_df['sku'] = price_df.Brand.replace({'APT': 'AP', 'C&G': 'CG'}) + price_df.Stage_acc.astype(str)

        date_list = price_df.columns[price_df.columns.astype(str).str.match(r'20\d{4}')].values.tolist()
        price_df = price_df.loc[:, ['SKU_wo_pkg', 'sku', 'Source counry', 'Brand', 'Tier', 'Stage_acc'] + date_list]
        col_mapping = {org: trans for org, trans in
                       zip(date_list, pd.to_datetime(pd.to_numeric(date_list, downcast='integer').astype(str),
                                                     format='%Y%m').strftime('%m-%d-%Y'))}
        col_mapping.update({'SKU_wo_pkg': 'sku_code', 'Source counry': 'country', 'Brand': 'brand', 'Tier': 'sub-brand',
                            'Stage_acc': 'stage'})
        price_df.rename(col_mapping, axis=1, inplace=True)

        # split the DataFrame into total price table and volume table
        vol_start = (price_df.country == 'Source counry').idxmax()
        vol_df = price_df.loc[vol_start + 1:, :]
        vol_df = vol_df[~vol_df.sku_code.isnull()]
        vol_df = vol_df.set_index(['sku_code', 'sku', 'country', 'brand', 'sub-brand', 'stage'])
        vol_df = vol_df.replace(r'\s+', '', regex=True).astype(float)

        price_df = price_df.loc[:vol_start - 1:, :]
        price_df = price_df[~price_df.sku_code.isnull()]
        price_df = price_df.set_index(['sku_code', 'sku', 'country', 'brand', 'sub-brand', 'stage'])
        price_df = price_df.replace(r'\s+', '', regex=True).astype(float)

        # Compute price per unit
        price_df /= vol_df

        price_df = price_df.stack().reset_index().rename({'level_6': 'date', 0: 'price'}, axis=1)

        price_df['source'] = 'smartpath'
        price_df['scope'] = 'DI'
        price_df = price_df.loc[:, ['sku_code', 'sku', 'country', 'brand', 'sub-brand', 'stage', 'source', 'scope',
                                    'date', 'price']]

        # added by ZX, make sure we have right date format
        if str(os.name) == 'nt':
            price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%#m/%#d/%Y')
        else:
            price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%-m/%-d/%Y')

        # price_df = price_df.set_index('sku_code') # set first column as index in case index is not dropped when
        # exported

        # output file
        self.output_file(price_df, file_id)

    def prep_sellin_eib(self, file_id: str = 'sellin_eib'):
        """
        The purpose of this method is to prepare EIB sellin data, to be used to calculate IL sellin
        """
        sellin_df = self.load_latest_file(file=file_id, skiprows=1)

        lookup_df = pd.read_csv(
            os.path.join(DIR_MAPPINGS, self.cfg[file_id]['brand_and_tier_mapping_sheet']),
            encoding='ISO-8859-1'
        )
        sku_mapping = pd.read_csv(os.path.join(DIR_MAPPINGS, self.cfg[file_id]['sku_mapping']), encoding='ISO-8859-1')

        tier_mapping = lookup_df.loc[:, ['Tier', 'Tier_acc']].dropna(axis=0)
        tier_mapping = tier_mapping[tier_mapping['Tier'] != tier_mapping['Tier_acc']].set_index('Tier').squeeze()

        sku_mapping = sku_mapping[~sku_mapping.Stage_acc.isnull()].drop_duplicates(['SKU No', 'Tier'])

        # Remove the columns with quarter info
        sellin_df = sellin_df.iloc[:, :np.argmax(['Unnamed' in col for col in sellin_df.columns[1:]]) + 1]

        sellin_df['scope'] = sellin_df.iloc[:, 0].astype(str).apply(lambda s: s.split()[0])
        sellin_df['scope'] = sellin_df['scope'].replace({'SC': 'EIB'})

        # Prepare SKU No and Tier for merge
        sellin_df['SKU No'] = pd.to_numeric(sellin_df['SKU No'], errors='coerce')
        sellin_df = sellin_df[~(sellin_df.iloc[:, 0].isnull() & sellin_df['SKU No'].isnull())]
        sellin_df['Tier'] = sellin_df['Tier'].replace(tier_mapping)

        sellin_df = sellin_df.merge(sku_mapping.loc[:, ['Tier_acc', 'SKU No', 'SKU_wo_pkg']], on=['SKU No'], how='left')

        # Drop rows with Tier conflicts
        sellin_df = sellin_df[sellin_df.Tier == sellin_df.Tier_acc]

        # define list of dates in the columns (5-digit strings)
        date_list = [s for s in sellin_df.columns if s.isdigit()]

        sellin_df.rename({'SKU_wo_pkg': 'sku_code'}, axis=1, inplace=True)
        sellin_df = sellin_df.loc[:, date_list + ['sku_code', 'scope']]
        sellin_df = sellin_df.set_index(['sku_code', 'scope'])
        sellin_df = sellin_df.stack().reset_index().rename({'level_2': 'date', 0: 'value'}, axis=1)
        sellin_df['date'] = pd.TimedeltaIndex(sellin_df['date'].astype(int), unit='d') + datetime(1900, 1, 1)
        sellin_df['date'] = [date_val.replace(day=1) for date_val in sellin_df['date']]

        sellin_df['status'] = 'actual'
        sellin_df.loc[sellin_df.date.dt.to_period('M') > pd.to_datetime('today').to_period('M'),
                      'status'] = 'forecasted'

        sellin_df['type'] = 'Sell-in'
        sellin_df['unit'] = 'ton'
        sellin_df['produced_date'] = 'Default'

        sellin_df = sellin_df.loc[:, ['sku_code', 'scope', 'type', 'date', 'produced_date', 'status', 'value', 'unit']]
        # sellin_df = sellin_df.set_index('sku_code')

        # output file
        self.eib_sellin = sellin_df
        # self.output_file(sellin_df, file_id)

    def prep_dc_anp(self, file='AnP'):
        """
        The purpose of this method is to prepare DC AnP data
        """
        # load file
        df = self.load_latest_file(file)

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # transform: melt multiple brand columns into one
        df = pd.melt(df, id_vars=['Date'],
                     value_vars=['AP', 'AC', 'NC', 'C&G', 'Karicare', 'Happy Family'])
        df.rename(columns={'variable': 'Brand', 'value': 'Spending'}, inplace=True)
        df['Date'] = pd.TimedeltaIndex(df['Date'], unit='d') + datetime(1900, 1, 1)

        # output file (with specified column order)
        df = df.loc[:, ['Date', 'Brand', 'Spending']]
        self.output_file(df, file)

    def prep_dc_osa(self, file='DC_OSA'):
        """
        The purpose of this method is to prepare DC OSA data
        """
        # load file
        df = self.load_latest_file(file)

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # output file (with specified column order)
        df = df.loc[:, ['SKU', 'Year', 'Month', 'OSA', 'CSL']]
        self.output_file(df, file)

    def prep_dc_store_dist(self, file='dc_store_dist'):
        """
        The purpose of this method is to check DC store distribution data
        """
        # load file
        df = self.load_latest_file(file)

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # output (with specified column order)
        df = df.loc[:, ['Month', 'AC', 'NC', 'AP']]
        self.output_file(df, file)

    def prep_productlist(self, file='productlist'):
        """
        The purpose of this method is to prepare master product list
        """
        # load file
        df = self.load_latest_file(file)

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # transformation: Rename columns to fit interface contract
        df.rename(columns={'SKU_type': 'SKU_Type',
                           'product_name': 'Name',
                           'SKU': 'Stage',
                           'brand': 'Brand',
                           'weight_per_tin': 'UnitWeight',
                           'price': 'UnitPrice',
                           'unit': 'Unit',
                           'unit_per_case': 'CaseUnit'},
                  inplace=True)
        # replace Chinese brand names
        df['Brand'] = df['Brand'].replace({'白金诺优能': 'NP',
                                           '可瑞康': 'KG',
                                           '诺优能': 'NC',
                                           '多美滋': 'DG',
                                           '爱他美': 'AC',
                                           '牛栏': 'CG',
                                           '白金爱他美': 'AP'})

        # output file (with specified column order)
        df = df.loc[:, ['SKU_NO', 'SKU_Type', 'Name', 'Brand', 'Stage', 'UnitPrice', 'UnitWeight', 'Unit', 'CaseUnit']]
        self.output_file(df, file)

    def prep_distributor(self, file='distributorlist'):
        """
        The purpose of this method is to prepare master distributor list
        """
        # load file
        df = self.load_latest_file(file)

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # transform: Rename columns to fit interface contract
        df.rename(columns={'Channel': 'Group'}, inplace=True)

        # output file (with specified column order)
        df = df.loc[:, ['SP_code', 'Name', 'Address', 'Group', 'SPGroup', 'SPSubGroup', 'Validity']]
        self.output_file(df, file)

    def prep_customers(self, file='customerlist'):
        """
        The purpose of this method is to prepare master customer table
        """
        # load file
        df = self.load_latest_file(file, encoding='utf_8')

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # transform: remove invalid stores
        df = df.loc[~df['store_name'].str.contains("QS虚拟门店"), :]

        # output file (with specified column order)
        df = df.loc[:, ['store_code', 'store_name', 'retailer',
                        'sub_region', 'channel', 'grade', 'cust_type', 'SP_code']]
        self.output_file(df, file, encoding='utf_8')

    def prep_pos(self, file='pos'):
        """
        The purpose of this method is to prepare DC historical offtake
        """
        # load file
        df = self.load_latest_file(file)

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # groupby unique index to avoid duplicates
        df = df.groupby(['date', 'store_code', 'SKU_NO']).sum().reset_index()

        # clean date format
        df['date'] = df['date'].astype(str) + '15'  # we add a day number so that it is read as a date
        if str(os.name) == 'nt':  # removing leading zeros
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%#m/%#d/%Y')
        else:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%-m/%-d/%Y')

        # output file (with specified column order)
        df = df.loc[:, ['date', 'store_code', 'SKU_NO', 'quantity', 'POS_value']]
        self.output_file(df, file)

    def prep_dms_sellout(self, file='dms_sellout'):
        """
        The purpose of this method is to prepare DC historical sellout data (from DMS)
        """
        # load file
        df = self.load_latest_file(file)

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # drop irrelevant columns
        df.drop('SP_code', axis=1, inplace=True)
        df.drop('customer_code', axis=1, inplace=True)
        # add required fields
        df['scope'] = 'DC'
        df['unit'] = 'TIN'
        df['type'] = 'sellout'
        # rename column per interface contract
        df = df.rename(columns={'SP_value': 'revenue'})
        # calculate price based on revenue/quantity
        df['price'] = df['revenue'] / df['quantity']

        # output file (with specified column order)
        df = df.loc[:, ['date', 'SKU_NO', 'scope', 'type', 'quantity', 'unit', 'price', 'SP_price', 'revenue']]
        self.output_file(df, file)

    def prep_sp_inv(self, file='sp_inv'):
        """
        The purpose of this method is to prepare service provide monthly inventory
        """
        # load file
        df = self.load_latest_file(file)
        df_sku = pd.read_csv(
            os.path.join(DIR_MAPPINGS, self.cfg[file]['mapping_file'])
        )

        # Check business contract for number of columns and column names
        self.business_contract_checks(df, file)

        # drop null SKU_NO
        df.dropna(axis=0, subset=['SKU_NO'], inplace=True)
        # add SKU_NO by two levels of matching (old and new)
        df = df.copy()  # create copy to change column type without warning
        df.loc[:, 'SKU_NO'] = df['SKU_NO'].apply(str)  # force string format
        df = pd.merge(df, df_sku[['SKU_NO', 'SKU']], on='SKU_NO', how='left')  # merge on 'SKU_NO'
        df_new_sku = df_sku[['SKU_NO_new', 'SKU']]  # keep only relevant columns from sku table
        df_new_sku = df_new_sku.copy()  # create copy to change column type without warning
        df_new_sku.loc[:, 'SKU_NO_new'] = df_new_sku['SKU_NO_new'].apply(str)  # force string format
        df = pd.merge(df, df_new_sku, left_on='SKU_NO', right_on='SKU_NO_new', how='left')
        df['SKU_x'] = df['SKU_x'].fillna(df['SKU_y'])
        df.drop('SKU_y', axis=1, inplace=True)
        df.rename(columns={'SKU_x': 'SKU'}, inplace=True)
        # add required columns
        df['scope'] = 'DC'
        df['unit'] = 'Tin'
        df['type'] = 'sp_inv'
        # drop irrelevant column
        df.drop('SKU_NO_new', axis=1, inplace=True)
        # rename column per interface contract
        df = df.rename(columns={'SP_value': 'revenue'})
        # calculate price
        df['price'] = df['revenue'] / df['quantity']
        # format date time
        if str(os.name) == 'nt':  # removing leading zeros in date format
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%#m/%#d/%Y')
        else:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%-m/%-d/%Y')

        # output file (with specified column order)
        df = df.loc[:, ['SP_code', 'date', 'SKU_NO', 'SKU', 'scope', 'type', 'quantity', 'unit', 'price', 'revenue']]
        self.output_file(df, file)

    def prep_di_tradeflow(self, file='di_tradeflow', bcg_sku_mapping='bcg_sku_mapping',
                          tin_ton_mapping='di_tin_to_ton_mapping') -> None:
        """
        The purpose of the function is to prepare di_tradeflow data (offtake, sellin, sellout, sp inv and retailer inv)
        for model inputs
        :param file: reference key of the data_prep.yml with data paths
        :param bcg_sku_mapping: reference key of the data_prep.yml with data paths (for sku mapping data)
        :param tin_ton_mapping: reference key of the data_prep.yml with data paths (for tin_ton mapping data)
        :return:
        """

        # Specify dates used to the filter the cycle data
        today = date.today()
        m = 1 - C_CYCLE_MONTH_LAG
        date_m_1 = datetime(today.year, today.month - m, 1)
        date_m_2 = datetime(today.year, today.month - (C_IL_RANGE - C_CYCLE_MONTH_LAG), 1)
        date_m_4_plan = datetime(today.year, today.month + (m + 3), 1)
        date_m_3_plan = datetime(today.year, today.month + (m + 2), 1)

        def prep_di_u1_sellout(_file='di_u1_sellout', mapping_file='di_customer_mapping_U1_sellout',
                               sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
            The purpose of this function is to read in u1_sellout data, and transformed it based on several
            user-specified business rules
            :param _file: reference key in the data_prep.yml, which contains the path and file name information
            :param mapping_file: reference key in the data_prep.yml linked to a csv. file used to map the customer
             channel information
            :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
            :return: transformed pandas dataframe
            """
            # load file
            latest_df = self.load_latest_file(_file)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)
            # added by ZX, log missing channel info
            non_matched_retailers = set(latest_df['客户']) - set(df_mapping['客户'])
            if len(non_matched_retailers):
                logger.warning(
                    f""" di u1 sellout retailers could not be mapped into channel: \n{non_matched_retailers}""")

            # transform
            # map customer channel information
            latest_df = pd.merge(latest_df, df_mapping[['客户', 'channel']], on='客户', how='left')
            # filter data based on user-specified business rules
            latest_df = latest_df.loc[latest_df['状态'] == '已发货']
            # select correct date col and standradize date format
            latest_df['date'] = pd.TimedeltaIndex(latest_df['U1系统发货日期'], unit='d') + datetime(1900, 1, 1)
            latest_df['date_for_matching'] = [date_val.replace(day=1) for date_val in latest_df['date']]
            # # select M-1 cycle data
            # latest_df = latest_df.loc[latest_df['date_for_matching'] == date_m_1]
            # modified by ZX, changed M-1 to M-2
            latest_df = latest_df.loc[latest_df['date_for_matching'] == date_m_2]
            # generate new columns
            latest_df['sp'] = 'U1'
            latest_df['type'] = 'sellout'
            latest_df['scope'] = 'DI'
            latest_df['status'] = 'actual'
            # use sku mapping table to standardize sku names
            latest_df = pd.merge(latest_df, df_sku_mapping[['U1 sellout/offtake', 'trade flow SKU desc']],
                                 left_on='SKU', right_on='U1 sellout/offtake', how='left')
            # select columns
            select_cols = ["trade flow SKU desc", "sp", "type", "channel", "scope", "status", "date_for_matching",
                           "销量"]
            latest_df = latest_df[select_cols]
            latest_df.columns = ["sku", "sp", "type", "channel", "scope", "status", "date", "quantity"]
            # added by ZX, after confirmation with Hans, assign default channel to non-mapped retailers
            latest_df.fillna(value={'channel': 'BBC'}, inplace=True)
            # aggregate volume at the specified granularity
            latest_df = latest_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            # output
            return latest_df

        def prep_di_u1_offtake(_file='di_u1_offtake', mapping_file='di_customer_mapping_U1_offtake',
                               sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in u1_offtake data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param mapping_file: reference key in the data_prep.yml linked to a csv. file used to map the customer
             channel information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            latest_df = self.load_latest_file(_file)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)
            non_matched_retailers = set(latest_df['Retailer']) - set(df_mapping['Retailer'])
            if len(non_matched_retailers):
                logger.warning(
                    f""" di u1 offtake retailers could not be mapped into channel: \n{non_matched_retailers}""")

            # transform
            # map customer channel information
            latest_df = pd.merge(latest_df, df_mapping[['Retailer', 'channel']], on='Retailer', how='left')
            # correct data format
            latest_df['date_for_matching'] = pd.to_datetime(latest_df[['Year', 'Month']].assign(DAY=1))
            # select M-1 and M-2 data
            latest_df = latest_df.loc[(latest_df['date_for_matching'] <= date_m_1)
                                      & (latest_df['date_for_matching'] >= date_m_2)]
            # add new columns
            latest_df['sp'] = 'U1'
            latest_df['type'] = 'offtake'
            latest_df['scope'] = 'DI'
            latest_df['status'] = 'actual'
            # standardize sku names
            latest_df = pd.merge(latest_df, df_sku_mapping[['U1 sellout/offtake', 'trade flow SKU desc']],
                                 left_on='U1 SKU', right_on='U1 sellout/offtake', how='left')
            # select columns
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "volume in tin"]
            latest_df = latest_df[select_cols]
            latest_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            # added by ZX, after confirmation with Hans, assign default value to non-mapped retailers
            latest_df.fillna(value={'channel': 'BBC'}, inplace=True)
            # aggregate volume by the specified granularity
            latest_df = latest_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            return latest_df

        def prep_di_yuou_sellout(_file='di_yuou_sellout', mapping_file='di_customer_mapping_yuou_sellout',
                                 sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in yuou_sellout data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param mapping_file: reference key in the data_prep.yml linked to a csv. file used to map the customer
             channel information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)
            # added by ZX, log missing channel info
            non_matched_retailers = set(transformed_df['客户']) - set(df_mapping['客户'])
            if len(non_matched_retailers):
                logger.warning(
                    f""" di yuou sell out retailers could not be mapped into channel: \n{non_matched_retailers}""")

            # transform
            # map customer channel information
            transformed_df = pd.merge(transformed_df, df_mapping[['客户', 'channel']], on='客户', how='left')
            # correct date format and select M-1 cycle data
            transformed_df['date'] = pd.TimedeltaIndex(
                transformed_df['渝欧系统发货日期'], unit='d'
            ) + datetime(1900, 1, 1)
            transformed_df['date_for_matching'] = [date_val.replace(day=1) for date_val in transformed_df['date']]
            # modified by ZX, change M-1 from M-2
            transformed_df = transformed_df.loc[pd.to_datetime(transformed_df['date_for_matching']) == date_m_2]
            # filtering based on user-specified business rules
            transformed_df = transformed_df.loc[transformed_df['状态'] == '已发货']
            # add new columns
            transformed_df['sp'] = 'Yuou'
            transformed_df['type'] = 'sellout'
            transformed_df['scope'] = 'DI'
            transformed_df['status'] = 'actual'
            # standardize sku names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['Yuou sellout/offtake', 'trade flow SKU desc']],
                                      left_on='SKU', right_on='Yuou sellout/offtake', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "销量"]
            transformed_df = transformed_df[select_cols]
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            transformed_df.fillna(value={'channel': 'Distribution'},
                                  inplace=True)  # added by ZX, after confirmation with Hans
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)['quantity'].agg('sum')
            return transformed_df

        def prep_di_yuou_offtake(
                _file='di_yuou_offtake', file_yunji='di_yuou_yunji_offtake',
                mapping_file='di_customer_mapping_yuou_sellout', sku_mapping='di_sku_mapping'
        ) -> pd.DataFrame:
            """
             The purpose of this function is to read in yuou_offtake data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param file_yunji: reference key in the data_prep.yml, which contains the path and file name information
             :param mapping_file: reference key in the data_prep.yml linked to a csv. file used to map the customer
             channel information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file)
            df_yunji = self.load_latest_file(file_yunji)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)
            # added by ZX, log missing channel info
            non_matched_retailers = set(transformed_df['客户']) - set(df_mapping['客户'])
            if len(non_matched_retailers):
                logger.warning(
                    f""" di yuou offtake retailers could not be mapped into channel: \n{non_matched_retailers}""")

            # transform
            # standardize date format and select M-1 and M-2 cycle data. should use datetime(1899, 12, 30)
            transformed_df['date'] = pd.TimedeltaIndex(transformed_df['渝欧系统发货日期'], unit='d') + datetime(1899, 12, 30)
            transformed_df['date_for_matching'] = [date_val.replace(day=1) for date_val in transformed_df['date']]
            transformed_df = transformed_df.loc[(transformed_df['date_for_matching'] <= date_m_1)
                                                & (transformed_df['date_for_matching'] >= date_m_2)]
            # should use datetime(1899, 12, 30)
            df_yunji['date'] = pd.TimedeltaIndex(df_yunji['渝欧系统发货日期'], unit='d') + datetime(1899, 12, 30)
            df_yunji['date_for_matching'] = [date_val.replace(day=1) for date_val in df_yunji['date']]
            df_yunji = df_yunji.loc[
                (df_yunji['date_for_matching'] <= date_m_1) & (df_yunji['date_for_matching'] >= date_m_2)]
            # filtering based on user-specified business rules
            transformed_df = transformed_df.loc[transformed_df['状态'] == '已发货']
            df_yunji = df_yunji.loc[df_yunji['状态'] == '已发货']
            # for customer == 'YUNJI' in transformed_df, replace the data from df_yunji
            transformed_df = transformed_df.loc[transformed_df['客户'] != 'YUNJI']
            transformed_df = pd.concat([transformed_df, df_yunji], ignore_index=True, sort=True)
            # map customer channel information
            transformed_df = pd.merge(transformed_df, df_mapping[['客户', 'channel']], on='客户', how='left')
            # add new columns
            transformed_df['sp'] = 'Yuou'
            transformed_df['type'] = 'offtake'
            transformed_df['scope'] = 'DI'
            transformed_df['status'] = 'actual'
            # standardize sku names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['Yuou sellout/offtake', 'trade flow SKU desc']],
                                      left_on='SKU', right_on='Yuou sellout/offtake', how='left')
            # select cols
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching", "销量"]
            transformed_df = transformed_df[select_cols]
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            # aggregate volume by specified granularity
            # added by ZX, after confirmation with Hans, assign default channel to non-mapped retailers
            transformed_df.fillna(value={'channel': 'Distribution'},
                                  inplace=True)
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)['quantity'].agg('sum')
            return transformed_df

        def prep_di_u1_sp_inv(_file='di_u1_sp_inventory', sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in u1_sp_inventory data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            # filtering data based on user-specified business rules
            transformed_df = transformed_df.loc[transformed_df['验收类型'] == '正品']
            transformed_df = transformed_df.loc[transformed_df['商户'] != 'pop']
            # add new columns
            transformed_df['sp'] = 'U1'
            transformed_df['type'] = 'sp_inv'
            transformed_df['scope'] = 'DI'
            transformed_df['status'] = 'actual'
            transformed_df['channel'] = 'Total'
            # modified by ZX, select M-2 cycle data
            transformed_df['date'] = date_m_2
            # standardize sku names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['U1 SP inv', 'trade flow SKU desc']],
                                      left_on='品牌SKU', right_on='U1 SP inv', how='left')
            # select cols
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date", "金蝶数量"]
            transformed_df = transformed_df[select_cols]
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            # aggregate volume based on specified granularity
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)['quantity'].agg('sum')
            return transformed_df

        def prep_di_yuou_sp_inv(_file='di_yuou_sp_inventory', sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in yuou_sp_inv data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            # modified by ZX, select M-2 cycle data
            transformed_df['date'] = date_m_2
            # filter data based on user-specified business rules
            transformed_df = transformed_df.loc[transformed_df['商户名称'].isin(['待销毁', '待销毁冻结', '废品', '残次品',
                                                                             '临期冻结'])]
            # add new columns
            transformed_df['sp'] = 'Yuou'
            transformed_df['type'] = 'sp_inv'
            transformed_df['scope'] = 'DI'
            transformed_df['status'] = 'actual'
            transformed_df['channel'] = 'Total'
            # standardize sku names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['Yuou SP inv', 'trade flow SKU desc']],
                                      left_on='达能SKU', right_on='Yuou SP inv', how='left')
            # select columns
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date", "数量"]
            transformed_df = transformed_df[select_cols]
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            # aggregate volume based on specified granularity
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)['quantity'].agg('sum')
            # output
            return transformed_df

        def prep_di_u1_retailer_inv(_file='di_u1_retailer_inventory',
                                    # mapping_file='di_customer_mapping_u1_retailer_inv',
                                    sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in u1_retailer_inv data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file)
            # df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            # standardize date format
            transformed_df = transformed_df[np.isfinite(transformed_df['月末库存1'])]
            transformed_df['date'] = pd.to_datetime([datetime.strptime(str(int(date_val)), '%Y%m')
                                                     for date_val in transformed_df['月末库存1']])
            # add new columns
            transformed_df['sp'] = 'U1'
            transformed_df['type'] = 'retailer_inv'
            transformed_df['scope'] = 'DI'
            transformed_df['status'] = 'actual'
            # map customer channel information
            # transformed_df = pd.merge(transformed_df, df_mapping[['客户名称', 'channel']], on='客户名称', how='left')
            transformed_df['channel'] = 'Total'
            # standardize sku names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['U1 retailer inv', 'trade flow SKU desc']],
                                      left_on='品牌-SKU', right_on='U1 retailer inv', how='left')
            # Modified by ZX, select cycle M-2 data
            transformed_df = transformed_df.loc[transformed_df['date'] == date_m_2]
            # select columns
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date", "安全库存数量"]
            transformed_df = transformed_df[select_cols]
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            # added by ZX, after confirmation with Hans, assign default channel to non-mapped retailers
            transformed_df.fillna(value={'channel': 'BBC'},
                                  inplace=True)
            # aggregate volume based on specified granularity
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)['quantity'].agg('sum')
            return transformed_df

        def prep_di_yuou_retailer_inv(_file='di_yuou_retailer_inventory', sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in yuou_retailer_inv data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            # Modifed by ZX, standardize date format and select M-2 cycle data
            transformed_df['date_for_matching'] = pd.to_datetime(
                [datetime.strptime(date_val, '%Y%m') for date_val in transformed_df['Month'].astype(str)])
            transformed_df = transformed_df.loc[transformed_df['date_for_matching'] == date_m_2]
            # add new columns
            transformed_df['channel'] = 'Total'
            transformed_df['sp'] = 'Yuou'
            transformed_df['type'] = 'retailer_inv'
            transformed_df['scope'] = 'DI'
            transformed_df['status'] = 'actual'
            # standardize sku names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['Yuou retailer inv (to be designed)',
                                                                      'trade flow SKU desc']],
                                      left_on='SKU', right_on='Yuou retailer inv (to be designed)', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "Volume"]
            transformed_df = transformed_df[select_cols]
            # transformed_df.loc[:, 'Volume'] = transformed_df.loc[:, 'Volume'].str.replace(',', '').astype(float)
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            # aggregate volume by specified granularity
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)[
                'quantity'].agg(
                'sum')

            return transformed_df

        def prep_di_u1_sellin(_file='di_u1_sellin', sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in yuou_retailer_inv data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file, encoding='unicode_escape')
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transformation
            # select columns indicating dates and other information
            df_dates_column_names = transformed_df.iloc[0, 5:].tolist()
            df_other_column_names = transformed_df.iloc[1, :5].tolist()
            transformed_df = transformed_df.iloc[2:, :]
            transformed_df.columns = df_other_column_names + df_dates_column_names
            # filter data based on user-specified business rules
            na_filter_cols = ["Plant", "Brand", "Code", "Description"]
            transformed_df.dropna(subset=na_filter_cols, how='all', inplace=True)  # filter totals
            # reshape the dataframe
            transformed_df = pd.melt(transformed_df, id_vars=df_other_column_names, value_vars=df_dates_column_names,
                                     var_name='date', value_name='volume')
            transformed_df = transformed_df.loc[transformed_df['Status'] == 'TTL']
            # standardize date format
            transformed_df['date'] = pd.TimedeltaIndex(
                transformed_df['date'].astype(int), unit='d'
            ) + datetime(1900, 1, 1)
            transformed_df['date_for_matching'] = [date_val.replace(day=1) for date_val in transformed_df['date']]
            # select columns
            transformed_df = transformed_df[['Brand', 'Plant', 'date_for_matching', 'volume']]

            # according to the user-specified business rules, select corresponding M+/- n cycle data
            # modified by ZX, change date_m_1 to date_m_2
            def define_status(row):
                if (row['Plant'] == 'Aintree') & (row['date_for_matching'] > date_m_2) & (
                        row['date_for_matching'] <= date_m_3_plan):
                    return 'forecasted'
                if (row['Plant'] != 'Aintree') & (row['date_for_matching'] > date_m_2) & (
                        row['date_for_matching'] <= date_m_4_plan):
                    return 'forecasted'
                if row['date_for_matching'] == date_m_2:
                    return 'actual'
                return 'Other'

            transformed_df['status'] = transformed_df.apply(lambda row: define_status(row), axis=1)
            transformed_df = transformed_df.loc[transformed_df['status'].isin(['forecasted', 'actual'])]
            # create new columns
            transformed_df['sp'] = 'U1'
            transformed_df['type'] = 'sellin'
            transformed_df['scope'] = 'DI'
            transformed_df['channel'] = 'Total'
            # standardize sku names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['sell in', 'trade flow SKU desc']],
                                      left_on='Brand', right_on='sell in', how='left')
            # select columns
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "volume"]
            transformed_df = transformed_df[select_cols]
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            transformed_df.loc[:, 'quantity'] = transformed_df.loc[:, 'quantity'].str.replace(',', '').astype(float)
            # aggregate volume by specified granualrity
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)[
                'quantity'].agg(
                'sum')
            return transformed_df

        def prep_di_yuou_sellin(_file='di_yuou_sellin', sku_mapping='di_sku_mapping') -> pd.DataFrame:
            """
             The purpose of this function is to read in yuou_sellin data, and transformed it based on several
             user-specified business rules
             :param _file: reference key in the data_prep.yml, which contains the path and file name information
             :param sku_mapping: reference key in the data_prep.yml linked to a csv. file used to standardize the
             sku names)
             :return: transformed pandas dataframe
             """
            # load file
            transformed_df = self.load_latest_file(_file, encoding='unicode_escape')
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transformation
            # select columns indicating dates and other useful information
            df_dates_column_names = transformed_df.iloc[0, 5:].tolist()
            df_other_column_names = transformed_df.iloc[1, :5].tolist()
            transformed_df = transformed_df.iloc[2:, :]
            transformed_df.columns = df_other_column_names + df_dates_column_names
            na_filter_cols = ["Plant", "Brand", "Code", "Description"]
            transformed_df.dropna(subset=na_filter_cols, how='all', inplace=True)  # filter totals
            # reshape dataframe
            transformed_df = pd.melt(transformed_df, id_vars=df_other_column_names, value_vars=df_dates_column_names,
                                     var_name='date',
                                     value_name='volume')
            transformed_df = transformed_df.loc[transformed_df['Status'] == 'TTL']
            # standradize date format
            transformed_df['date'] = pd.TimedeltaIndex(
                transformed_df['date'].astype(int), unit='d'
            ) + datetime(1900, 1, 1)
            transformed_df['date_for_matching'] = [date_val.replace(day=1) for date_val in transformed_df['date']]
            transformed_df = transformed_df[['Brand', 'Plant', 'date_for_matching', 'volume']]

            # based on the user-defined business rules, select M+/-n cycle data
            # modified by ZX, change date_m_1 to date_m_2
            def define_status(row):
                if (row['Plant'] == 'Aintree') & (row['date_for_matching'] > date_m_2) & (
                        row['date_for_matching'] <= date_m_3_plan):
                    return 'forecasted'
                if (row['Plant'] != 'Aintree') & (row['date_for_matching'] > date_m_2) & (
                        row['date_for_matching'] <= date_m_4_plan):
                    return 'forecasted'
                if row['date_for_matching'] <= date_m_2:
                    return 'actual'
                return 'Other'

            transformed_df['status'] = transformed_df.apply(lambda row: define_status(row), axis=1)
            transformed_df = transformed_df.loc[transformed_df['status'].isin(['forecasted', 'actual'])]
            # create new columns
            transformed_df['sp'] = 'Yuou'
            transformed_df['type'] = 'sellin'
            transformed_df['scope'] = 'DI'
            transformed_df['channel'] = 'Total'
            # standardize column names
            transformed_df = pd.merge(transformed_df, df_sku_mapping[['sell in', 'trade flow SKU desc']],
                                      left_on='Brand', right_on='sell in', how='left')
            # select columns
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "volume"]
            transformed_df = transformed_df[select_cols]
            transformed_df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            transformed_df.loc[:, 'quantity'] = transformed_df.loc[:, 'quantity'].str.replace(',', '').astype(float)
            # aggregate volume based on defined granularity
            transformed_df = transformed_df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"],
                                                    as_index=False)[
                'quantity'].agg('sum')
            return transformed_df

        # use the functions defined above to load and transform all di related input data
        df_u1_sellout = prep_di_u1_sellout()
        df_u1_offtake = prep_di_u1_offtake()
        df_yuou_sellout = prep_di_yuou_sellout()
        df_yuou_offtake = prep_di_yuou_offtake()
        df_u1_sp_inv = prep_di_u1_sp_inv()
        df_yuou_sp_inv = prep_di_yuou_sp_inv()
        df_u1_retailer_inv = prep_di_u1_retailer_inv()
        df_yuou_retailer_inv = prep_di_yuou_retailer_inv()
        df_u1_sellin = prep_di_u1_sellin()
        df_yuou_sellin = prep_di_yuou_sellin()

        # concatenate all di data
        # Changed by ZX, confirmed with Hao and Serena, remove duplicated use of df_u1_retailer_inv
        df = pd.concat(
            [df_u1_offtake, df_u1_retailer_inv, df_u1_sellout, df_u1_sp_inv, df_yuou_offtake, df_yuou_sellout,
             df_yuou_sp_inv, df_yuou_retailer_inv, df_u1_sellin, df_yuou_sellin],
            ignore_index=True)

        # map di sku to BCG sku
        bcg_sku_mapping = self.load_file(bcg_sku_mapping)
        bcg_sku_mapping = bcg_sku_mapping.loc[bcg_sku_mapping['Is_Mature'] == True, :]
        df = pd.merge(df,
                      bcg_sku_mapping[['SKU', 'Country', 'Brand_acc', 'Tier_acc', 'Stage_acc', 'Package_acc', 'SKU_std',
                                       'SKU_wo_pkg']],
                      left_on='sku', right_on='SKU', how='left')

        # tin to ton
        tin_ton_mapping = self.load_latest_file(tin_ton_mapping)
        df = pd.merge(df, tin_ton_mapping[['SKU', 'Weight (gram) after downsize']], left_on='sku', right_on='SKU',
                      how='left')
        df['quantity_ton'] = df['quantity'] * df['Weight (gram) after downsize'] / 1000000

        # output
        df = df.rename(columns={'Country': 'country', 'Brand_acc': 'brand', 'Tier_acc': 'tier', 'Stage_acc': 'stage',
                                'Package_acc': 'package'})
        df['unit'] = 'ton'
        df.replace({'channel': {'ALI': 'Ali',
                                'New channel': 'NewChannel'}}, inplace=True)

        # format date
        if str(os.name) == 'nt':  # check with os is windows
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%#m/%#d/%Y')
            df['produced_date'] = today.strftime('%#m/%#d/%Y')
        else:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%-m/%-d/%Y')
            df['produced_date'] = today.strftime('%-m/%-d/%Y')

        df = df[
            ['SKU_std', 'SKU_wo_pkg', 'sku', 'country', 'brand', 'tier', 'stage', 'package', 'sp', 'channel', 'type',
             'scope', 'status', 'date', 'produced_date', 'quantity_ton', 'unit']]
        df = df.rename(columns={'quantity_ton': 'quantity',
                                'SKU_std': 'sku_code'})
        self.di_tradeflow = df
        self.output_file(df, file)

    def prep_il_sellin(self, file='il_sellin'):
        """
        The purpose of this method is to calculate IL sellin based on DI and EIB
        """
        today = date.today()
        date_m_2 = datetime(today.year, today.month - (C_IL_RANGE - C_CYCLE_MONTH_LAG), 1)
        df_di = self.di_tradeflow  # load DI data from object
        df_eib = self.eib_sellin  # load EIB data from object

        # prepare DI data
        df_di = df_di.loc[df_di['type'] == 'sellin', :]  # select sellin data only
        df_di = df_di.loc[:, ['SKU_wo_pkg', 'status', 'date', 'quantity']]  # select relevant columns
        df_di = df_di.groupby(['SKU_wo_pkg', 'status', 'date']).sum().reset_index()  # remove dup with groupby+sum
        df_di.rename(columns={'quantity': 'DI', 'SKU_wo_pkg': 'sku_code'}, inplace=True)  # rename columns
        df_di['date'] = pd.to_datetime(df_di['date'])  # enforce date format

        # prepare EIB data
        df_eib = df_eib.loc[df_eib['scope'] == 'EIB', :]  # select sellin data only
        df_eib = df_eib.loc[:, ['sku_code', 'date', 'status','value']]  # select relevant columns
        df_eib.rename(columns={'value': 'EIB',
                               'status': 'eib_status'}, inplace=True)  # rename columns
        df_eib['date'] = pd.to_datetime(df_eib['date'])  # enforce date format

        # merge DI and EIB data
        df_il_sellin = pd.merge(df_di,
                                df_eib,
                                how='outer',
                                on=['date', 'sku_code'])  # merge DI and EIB sellin data
        df_il_sellin.fillna({'DI': 0, 'EIB': 0}, inplace=True)  # fill null values with 0
        df_il_sellin['status'] = df_il_sellin['status'].fillna(df_il_sellin['eib_status'])
        df_il_sellin.drop('eib_status', axis=1, inplace=True)
        df_il_sellin['IL'] = df_il_sellin['DI'] + df_il_sellin['EIB']  # calculate IL sellin = DI + EIB
        df_il_sellin = pd.melt(df_il_sellin,
                               id_vars=['date', 'sku_code', 'status'],
                               value_vars=['DI', 'EIB', 'IL'])  # melt IL, DI, EIB columns into one colume
        df_il_sellin['value'] = df_il_sellin['value'].fillna(0)  # fill null value with 0
        df_il_sellin.rename(columns={'variable': 'scope'}, inplace=True)  # rename column
        df_il_sellin['type'] = 'Sell-in'
        df_il_sellin['unit'] = 'ton'

        df_il_sellin = df_il_sellin.loc[df_il_sellin['date'] >= date_m_2, :]
        df_il_sellin = df_il_sellin.loc[df_il_sellin['date'] < pd.to_datetime(today), :]

        # df_il_sellin.loc[
        #     (df_il_sellin['status'] == 'actual')
        # ].groupby('date').agg({'value': np.sum})

        # format date
        if str(os.name) == 'nt':  # check with os is windows
            df_il_sellin['date'] = pd.to_datetime(df_il_sellin['date']).dt.strftime('%#m/%#d/%Y')
            df_il_sellin['produced_date'] = today.strftime('%#m/%#d/%Y')
        else:
            df_il_sellin['date'] = pd.to_datetime(df_il_sellin['date']).dt.strftime('%-m/%-d/%Y')
            df_il_sellin['produced_date'] = today.strftime('%-m/%-d/%Y')

        # select and order columns
        df_il_sellin = df_il_sellin.loc[:, ['sku_code', 'scope', 'type',
                                            'date', 'produced_date', 'status', 'value', 'unit']]

        # export output
        self.output_file(df_il_sellin, file)

    def prep_il_offtake(self, file='il_offtake') -> None:
        """
        The purpose of this function is to prepare il offtake data for the model inputs
        :param file: reference key of data_prep.yml for data paths
        :return: None
        """
        today = date.today()
        date_m_2 = datetime(today.year, today.month - (C_IL_RANGE - C_CYCLE_MONTH_LAG), 1)

        # step 1: loading data at channel*brand*country level
        def il_automation_load_raw_data(data_path, row_mapping_path) -> pd.DataFrame:
            """
            The purpose of this function is to load all input data at brand level from all channels (except OSW)
            :param data_path: reference key in the data_prep.yml, which contains the path and file name information
            :param row_mapping_path: reference key in the data_prep.yml, and the reference mapping file indicating the
            rows with brands that should be selected from each raw input file
            :return: transformed input data as a pandas dataframe
            """
            il_offtake = self.load_latest_file(data_path, encoding='ISO-8859-1', skip_blank_lines=False,
                                               header=None)
            il_offtake_row_mapping = self.load_latest_file(row_mapping_path, encoding='ISO-8859-1',
                                                           skip_blank_lines=False,
                                                           header=None)

            # Search for the absolute volume values
            abs_starts_at = np.argwhere((il_offtake.iloc[1, :] == 'Absolute').values)[-1][0]
            il_offtake = il_offtake.iloc[:, abs_starts_at:]

            # Convert dates
            il_offtake.columns = pd.to_datetime(il_offtake.iloc[2, :].astype(int).astype(str), format='%y%m')

            # Select only the relevant rows
            il_offtake_row_mapping.dropna(axis=0, how='all', inplace=True)
            il_offtake_row_mapping = il_offtake_row_mapping.T.set_index(0).T.set_index('row_no')
            il_offtake_row_mapping = il_offtake_row_mapping.loc[:, ['Full_name', 'Country', 'Channel']]
            il_offtake_row_mapping.index = il_offtake_row_mapping.index.astype(int) - 1

            il_offtake = il_offtake.join(il_offtake_row_mapping, how='right')
            il_offtake.dropna(axis=1, how='all', inplace=True)

            # Reshaping
            il_offtake.rename({'Full_name': 'brand'}, axis=1, inplace=True)
            il_offtake = il_offtake.set_index(['brand', 'Country', 'Channel'])
            il_offtake.columns = il_offtake.columns.rename('date')
            il_offtake = il_offtake.stack().reset_index(name='value')

            il_offtake.value = pd.to_numeric(il_offtake.value, errors='coerce').fillna(0)
            il_offtake.columns = il_offtake.columns.str.lower()

            return il_offtake

        def il_automation_load_raw_data_osw(data_path, row_mapping_path) -> pd.DataFrame:
            """
            The purpose of this function is to load all input data at brand level from OSW channels
            :param data_path: reference key in the data_prep.yml, which contains the path and file name information
            :param row_mapping_path: reference key in the data_prep.yml, and the reference mapping file indicating the
            rows with brands that should be selected from each raw input file
            :return: transformed input data as a pandas dataframe
            """

            # load raw files
            il_offtake = self.load_latest_file(data_path, encoding='utf-8', skip_blank_lines=False, header=None)
            il_offtake_row_mapping = self.load_file(row_mapping_path, encoding='utf-8', skip_blank_lines=False,
                                                    header=None)
            # Remove 1st column if totally empty
            if il_offtake.iloc[:, 0].isnull().all():
                il_offtake = il_offtake.iloc[:, 1:]

            # Find the row with the dates
            number_dates_per_row = [(i, (~pd.to_datetime(row, errors='coerce').isnull()).sum()) for i, row in
                                    enumerate(il_offtake.itertuples())]
            date_row = max(number_dates_per_row, key=lambda t: t[1])[0]
            # data ends at the 1st empty cell in the date row (or at the last row)
            data_ends_at = il_offtake.iloc[date_row, 1:].isnull().values.argmax() + 1
            if data_ends_at == 1:
                data_ends_at = il_offtake.shape[1]
            il_offtake = il_offtake.iloc[:, 1:data_ends_at]

            il_offtake.columns = il_offtake.iloc[date_row, :]

            # Select only the relevant rows
            il_offtake_row_mapping.dropna(axis=0, how='all', inplace=True)
            il_offtake_row_mapping = il_offtake_row_mapping.T.set_index(0).T.set_index('row_no')
            il_offtake_row_mapping = il_offtake_row_mapping.loc[:, ['Full_name', 'Country', 'Channel']]
            il_offtake_row_mapping.index = il_offtake_row_mapping.index.astype(int) - 1

            il_offtake = il_offtake.join(il_offtake_row_mapping, how='right')
            il_offtake.dropna(axis=1, how='all', inplace=True)

            # Reshaping
            il_offtake.rename({'Full_name': 'brand'}, axis=1, inplace=True)
            il_offtake = il_offtake.set_index(['brand', 'Country', 'Channel'])

            il_offtake.columns = il_offtake.columns.rename('date')
            il_offtake = il_offtake.stack().reset_index(name='value')
            il_offtake.columns = il_offtake.columns.str.lower()

            il_offtake['date'] = pd.TimedeltaIndex(il_offtake['date'].astype(int), unit='d') + datetime(1900, 1, 1)
            il_offtake['date'] = [date_val.replace(day=1) for date_val in il_offtake['date']]

            return il_offtake

        def combine_df(list_of_df: list) -> pd.DataFrame:
            """
            The purpose of this function is to combine all brand level data
            :param list_of_df: list of dataframes
            :return: a dataframe containing brand level data from all channels
            """
            combined_df = pd.concat(list_of_df)

            # rename brand to acc. names
            combined_df.loc[combined_df['brand'].str.contains('Aptamil'), 'brand'] = "APT"
            combined_df.loc[combined_df['brand'].str.contains('Karicare'), 'brand'] = "KC"
            combined_df.loc[combined_df['brand'].str.contains('Cow & Gate'), 'brand'] = "C&G"
            combined_df.loc[combined_df['brand'].str.contains('Nutrilon'), 'brand'] = "NC"

            # select data from month m_2
            # combined_df = combined_df.loc[combined_df.date >= date_m_2]

            return combined_df

        # step 2: loading data at sku level
        def il_automation_load_wechat_sku(_file) -> pd.DataFrame:
            """
            The purpose of this function is to load all input data at sku level from EC channel
            :param _file: reference key in the data_prep.yml, which contains the path and file name information
            :return: transformed input data as a pandas dataframe
            """

            # load the raw data
            il_offtake = self.load_latest_file(_file, encoding='utf-8', skip_blank_lines=True,
                                               header=None)

            # Remove 1st column if totally empty
            if il_offtake.iloc[:, 0].isnull().all():
                il_offtake = il_offtake.iloc[:, 1:]

            # Find the row and column with volume
            volume_row = 0
            for i, row in il_offtake.iterrows():
                if 'Sales Volume(Ton)' in row.tolist():
                    volume_row = i + 2
                    break

            volume_col_end = 0
            for i, row in il_offtake.iterrows():
                if 'Value Share' in row.tolist():
                    volume_col_end = row.tolist().index("Value Share") - 1
                    break

            # Find the row defining column names
            col_names_row = 0
            for i, row in il_offtake.iterrows():
                if 'Country' in row.tolist():
                    col_names_row = i
                    break

            col_names = il_offtake.iloc[col_names_row, :volume_col_end].tolist()

            # Select rows starting from volume_row
            il_offtake = il_offtake.iloc[volume_row:, :volume_col_end]

            il_offtake.columns = col_names

            # Reshaping
            cols_non_datetime = ['Country', 'Description', 'EN Description']
            cols_datetime = [name for name in col_names if name not in cols_non_datetime]
            il_offtake = pd.melt(il_offtake, id_vars=cols_non_datetime, value_vars=cols_datetime, var_name='date',
                                 value_name='volume')

            # fix date format
            il_offtake['date'] = pd.to_datetime(
                [datetime.strptime(date_val, '%Y%m.0') for date_val in il_offtake['date'].astype(str)])

            il_offtake = il_offtake.rename(columns={'Description': 'cn description'})
            il_offtake = il_offtake[['cn description', 'date', 'volume']]
            il_offtake['channel'] = 'wechat'

            return il_offtake

        def il_automation_load_ec_sku(_file) -> pd.DataFrame:
            """
            The purpose of this function is to load all input data at sku level from Wechat channel
            :param _file: reference key in the data_prep.yml, which contains the path and file name information
            :return: transformed input data as a pandas dataframe
            """
            # load raw data
            il_offtake = self.load_latest_file(_file, encoding='utf-8', skip_blank_lines=True, header=None)

            # Remove column if totally empty
            if il_offtake.iloc[:, 1].isnull().all():
                il_offtake = il_offtake.iloc[:, 1:]

            # Find the row defining column names
            col_names_row = 0
            for i, row in il_offtake.iterrows():
                if 'Source counry' in row.tolist():
                    col_names_row = i
                    break

            col_names = il_offtake.iloc[col_names_row, :].tolist()
            col_names = [name for name in col_names if str(name) != 'nan']

            # if the 'val' is in col_names, change it to volume
            col_names = ['CN Description' if name == "Sales Val'000 RMB" else name for name in col_names]

            # Find the row with volume
            volume_row = 0
            for i, row in il_offtake.iterrows():
                if 'Sales Vol(ton)' in row.tolist():
                    volume_row = i + 1
                    break

                    # Select rows starting from volume_row
            il_offtake = il_offtake.iloc[volume_row:, :len(col_names)]
            il_offtake.columns = col_names

            # Reshaping
            cols_non_datetime = ['Source counry', 'Brand', 'Tier', 'Stage', 'SKU', 'EN Description', 'CN Description']
            cols_datetime = [name for name in col_names if name not in cols_non_datetime]
            il_offtake = pd.melt(il_offtake, id_vars=cols_non_datetime, value_vars=cols_datetime, var_name='date',
                                 value_name='volume')

            # fix date format
            il_offtake['date'] = pd.to_datetime(
                [datetime.strptime(date_val, '%Y%m.0') for date_val in il_offtake['date'].astype(str)])

            il_offtake = il_offtake[['CN Description', 'date', 'volume']]
            il_offtake['channel'] = 'EC'
            il_offtake.rename(columns={'CN Description': 'cn description'}, inplace=True)

            return il_offtake

        def combine_sku_data(list_of_df: list, mapping_data='il_sku_database') -> pd.DataFrame:
            """
            The purpose of this function is to concatenate all sku level data from EC, Wechat and OSW channels
            :param list_of_df: list of dataframes
            :param mapping_data: reference key in the data_prep.yml, which link to the path of a mapping file
            that can be used to standardize sku names
            :return: transformed sku level data from all channels as a pandas dataframe
            """

            # loading
            df_sku_concat = pd.concat(list_of_df)
            df_sku_mapping = self.load_file(mapping_data)
            df_sku_mapping = df_sku_mapping.query('Group == "IL"')[['SKU', 'Country', 'Brand_acc', 'Description CN']]
            df_sku_mapping = df_sku_mapping.rename(columns={'Description CN': 'cn description', 'Brand_acc': 'brand'})
            df_sku_mapping.columns = df_sku_mapping.columns.str.lower()

            # rename CN description for mapping
            df_sku_concat['cn description'] = df_sku_concat['cn description'].replace({
                # KC ANZ (Australia + New Zealand)
                '澳大利亚金装可瑞康1段': '金装可瑞康1段',
                '新西兰金装可瑞康1段': '金装可瑞康1段',
                '澳大利亚金装可瑞康2段': '金装可瑞康2段',
                '新西兰金装可瑞康2段': '金装可瑞康2段',
                '澳大利亚金装可瑞康3段': '金装可瑞康3段',
                '新西兰金装可瑞康3段': '金装可瑞康3段',
                '澳大利亚金装可瑞康4段': '金装可瑞康4段',
                '新西兰金装可瑞康4段': '金装可瑞康4段',
                # AC DE
                '德国爱他美1+': '德国爱他美4段 (1岁以上)',
                '德国爱他美2+': '德国爱他美5段 (2岁以上)',
                '德国爱他美pre段': '德国爱他美PRE段',
                # AC PF ANZ
                '澳洲爱他美白金版1段': '澳洲爱他美白金1段',
                '澳洲爱他美白金版2段': '澳洲爱他美白金2段',
                '澳洲爱他美白金版3段': '澳洲爱他美白金3段',
                '澳洲爱他美白金版4段': '澳洲爱他美白金4段',
                # AC PN ANZ
                '澳洲爱他美金装1段': '澳洲金装爱他美1段',
                '澳洲爱他美金装2段': '澳洲金装爱他美2段',
                '澳洲爱他美金装3段': '澳洲金装爱他美3段',
                '澳洲爱他美金装4段': '澳洲金装爱他美4段',
                # AC UK
                '英国爱他美 1段': '英国爱他美1段',
                '英国爱他美 2段': '英国爱他美2段',
                '英国爱他美 3段': '英国爱他美3段 (1岁以上)',
                '英国爱他美 4段': '英国爱他美4段 (2岁以上)',
                # AC PF UK
                '英国爱他美白金1段': '英国爱他美白金版1段',
                '英国爱他美白金2段': '英国爱他美白金版2段',
                '英国爱他美白金3段': '英国爱他美白金版3段',
                # C&G UK
                '英国牛栏3段': '英国牛栏3段 (1岁以上)',
                '英国牛栏4段': '英国牛栏4段 (2岁以上)'})

            # get tier and stage info from sku info
            def get_tier(_df):
                sku = _df['sku']
                tier = ''
                list_tiers = ['PN', 'PF', 'COW', 'GOAT', 'C&G']
                for elem in list_tiers:
                    if elem in sku:
                        tier = elem
                return tier

            def get_stage(_df):
                sku = _df['sku']
                stage = ''
                list_stages = ['PRE', '1', '2', '3', '4', '5', '6']
                for elem in list_stages:
                    if elem in sku:
                        stage = elem
                return stage

            def get_sku_split(_df_sku, _df_sku_mapping):
                _df_sku = pd.merge(_df_sku, _df_sku_mapping, on='cn description')
                _df_sku['tier'] = _df_sku.apply(get_tier, axis=1)
                _df_sku['stage'] = _df_sku.apply(get_stage, axis=1)
                _df_sku.columns = _df_sku.columns.str.lower()
                _df_sku['volume'] = pd.to_numeric(_df_sku.volume, errors='coerce')
                return _df_sku

            df_sku_concat = get_sku_split(df_sku_concat, df_sku_mapping)

            # select only last month data
            # df_sku_concat = df_sku_concat.loc[df_sku_concat.date >= date_m_2]

            return df_sku_concat

        # step 3: get tier and stage level % from df_sku and apply them to all skus in df

        def cal_sku_volume(_df, _df_sku) -> pd.DataFrame:
            """
            The purpose of this function is to calcuate the volume % at SKU level, and apply the % to all data at
            brand level
            :param _df: a pandas dataframe of data at brand level
            :param _df_sku: a pandas dataframe of data at sku level
            :return: data at sku level with sku % as a pandas dataframe
            """
            # aggregate volume at sku level (i.e., country*brand*tier*stage)
            volume_tier = _df_sku.groupby(['country', 'brand', 'tier', 'stage', 'date'],
                                          as_index=False)['volume'].agg('sum')
            # aggregate volume at brand level
            volume_brand = _df_sku.groupby(['country', 'brand', 'date'], as_index=False)['volume'].agg('sum')
            # calculate sku volume split %
            df_ratio = pd.merge(volume_tier, volume_brand, on=['country', 'brand', 'date'],
                                suffixes=['_tier_stage', '_brand'])
            df_ratio['ratio_brand'] = df_ratio['volume_tier_stage'] / df_ratio['volume_brand']
            # apply the calcuated % to all brands from all the channels
            _df = pd.merge(_df, df_ratio[['country', 'brand', 'tier', 'stage', 'date', 'ratio_brand']])
            # select columns
            _df['volume'] = _df['value'].astype(float) * _df['ratio_brand']
            _df = _df[['date', 'country', 'brand', 'tier', 'stage', 'channel', 'volume']]
            # rename selected tier names for standardization
            _df.loc[_df['tier'] == 'COW', 'tier'] = 'GD'
            _df.loc[_df['tier'] == 'GOAT', 'tier'] = 'GT'
            # create new columns
            _df['sku_code'] = _df['country'] + '_' + _df['brand'] + '_' + _df['tier'] + '_' + _df['stage']
            _df['scope'] = 'IL'
            _df['type'] = 'offtake'
            _df['unit'] = 'ton'
            _df['produced_date'] = today
            _df['status'] = 'actual'
            # select columns
            _df = _df[
                ['sku_code', 'scope', 'country', 'brand', 'tier', 'stage', 'type', 'date', 'produced_date', 'status',
                 'volume', 'channel', 'unit']]

            return _df

        bc = il_automation_load_raw_data('il_bchannel', 'il_row_mapping_bchannel')
        cc = il_automation_load_raw_data('il_cchannel', 'il_row_mapping_cchannel')
        ofs = il_automation_load_raw_data('il_ofs', 'il_row_mapping_ofs')
        wc = il_automation_load_raw_data('il_wc', 'il_row_mapping_wc')
        ff = il_automation_load_raw_data('il_ff', 'il_row_mapping_ff')
        o2o = il_automation_load_raw_data('il_o2o', 'il_row_mapping_o2o')
        pdd = il_automation_load_raw_data('il_pdd', 'il_row_mapping_pdd')
        osw_de_all = il_automation_load_raw_data_osw('il_osw_de', 'il_row_mapping_osw_de')
        osw_anz = il_automation_load_raw_data_osw('il_osw_anz', 'il_row_mapping_osw_anz')
        osw_nl_all = il_automation_load_raw_data_osw('il_osw_nl', 'il_row_mapping_osw_nl')
        osw_nl = osw_nl_all.loc[osw_nl_all.brand == 'Nutrilon NL']
        osw_de = osw_de_all.loc[osw_de_all.brand == 'Aptamil DE']
        df = combine_df([bc, cc, ofs, wc, ff, o2o, pdd, osw_de, osw_anz, osw_nl])

        wechat_sku = il_automation_load_wechat_sku('il_sku_wechat')
        ec_sku = il_automation_load_ec_sku('il_sku_ec')
        osw_de_sku = osw_de_all.loc[osw_de_all.brand != 'Aptamil DE']
        osw_de_sku = osw_de_sku.rename(columns={'brand': 'cn description', 'Channel': 'channel', 'value': 'volume'})
        osw_de_sku = osw_de_sku[['cn description', 'date', 'volume', 'channel']]
        osw_nl_sku = osw_nl_all.loc[osw_nl_all.brand != 'Nutrilon NL']
        osw_nl_sku = osw_nl_sku.rename(columns={'brand': 'cn description', 'Channel': 'channel', 'value': 'volume'})
        osw_nl_sku = osw_nl_sku[['cn description', 'date', 'volume', 'channel']]
        df_sku = combine_sku_data([wechat_sku, ec_sku, osw_de_sku, osw_nl_sku])

        df = cal_sku_volume(df, df_sku)
        # changed by ZX, actuals should only till M-2
        # df = df.loc[df.date <= pd.to_datetime(today), :]  # keep actuals only
        df = df.loc[df.date <= date_m_2, :]

        # output files
        df = df[['sku_code', 'scope', 'type', 'date', 'produced_date', 'status', 'volume', 'channel', 'unit']]

        # added by ZX, make sure we have right date format
        if str(os.name) == 'nt':
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%#m/%#d/%Y')
        else:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%-m/%-d/%Y')
        if str(os.name) == 'nt':
            df['produced_date'] = pd.to_datetime(df['produced_date']).dt.strftime('%#m/%#d/%Y')
        else:
            df['produced_date'] = pd.to_datetime(df['produced_date']).dt.strftime('%-m/%-d/%Y')

        self.il_offtake = df
        self.output_file(df, file)

    def prep_eib_offtake(self, file_eib='eib_offtake') -> None:
        """
        The purpose of this function is to calculate eib offtake volume from the il offtake and di offtake
        :param file_eib: reference key of the data_prep.yml, which indicates the data paths of eib_offtake
        :return: None
        """
        today = date.today()
        date_m_2 = datetime(today.year, today.month - (C_IL_RANGE - C_CYCLE_MONTH_LAG), 1)

        # load il offtake data
        df_il_offtake = self.il_offtake

        # load di tradeflow data
        df_di_tradeflow_new = self.di_tradeflow
        df_di_tradeflow_new.columns = [c.lower() for c in df_di_tradeflow_new.columns]
        df_di_tradeflow_new['date'] = pd.to_datetime(df_di_tradeflow_new['date'])
        df_di_tradeflow_new['produced_date'] = pd.to_datetime(df_di_tradeflow_new['produced_date'])

        # load DI data from Impala to get full history
        df_di_tradeflow_old = get_latest_di_tradeflow()
        # keep the new records from df_di_tradeflow_new when existing in df_di_tradeflow_old
        df_di_tradeflow = pd.concat([df_di_tradeflow_new, df_di_tradeflow_old])
        # Modify the in-consistent naming for same channel
        df_di_tradeflow['channel'] = df_di_tradeflow['channel'].str.replace('ALI', 'Ali')
        df_di_tradeflow['channel'] = df_di_tradeflow['channel'].str.replace('New channel', 'NewChannel')
        df_di_tradeflow.drop_duplicates(
            ['sku_code', 'sku_wo_pkg', 'sku', 'country', 'brand', 'tier', 'stage',
             'package',
             # 'sp', # There're multiple sps for same sku_date_channel
             'channel', 'type', 'scope', 'status', 'date',
             # 'produced_date',# There're multiple produced_date for same sku_date_channel
             'unit'],
            keep='first',  # we keep the newest value
            inplace=True
        )

        # transformation
        # select offtake data at sku level
        df_di_offtake = df_di_tradeflow.loc[df_di_tradeflow['type'] == 'offtake']
        df_di_offtake = df_di_offtake.loc[df_di_offtake['channel'] != 'Total']
        # Filter forecasted values
        df_di_offtake = df_di_offtake[df_di_offtake.status == 'actual']
        df_di_offtake.to_csv('./.cache/di_tradeflow.csv')
        # df_di_offtake_by_sku = df_di_offtake.groupby(['date', 'SKU_wo_pkg'], as_index=False)['quantity'].agg('sum')
        df_di_offtake_by_sku = df_di_offtake.groupby(['date', 'sku_wo_pkg'], as_index=False)['quantity'].agg('sum')
        df_di_offtake_by_sku['date'] = pd.to_datetime(df_di_offtake_by_sku.date)
        # df_di_offtake_by_sku = df_di_offtake_by_sku.loc[df_di_offtake_by_sku['date'] >= date_m_2, :]

        df_il_offtake_by_sku = df_il_offtake.groupby(['date', 'sku_code'], as_index=False)['volume'].agg('sum')
        df_il_offtake_by_sku['date'] = pd.to_datetime(df_il_offtake_by_sku.date)
        df_all = pd.merge(df_il_offtake_by_sku,
                          df_di_offtake_by_sku,
                          # left_on=['sku_code', 'date'], right_on=['SKU_wo_pkg', 'date'],
                          left_on=['sku_code', 'date'], right_on=['sku_wo_pkg', 'date'],
                          how='left')
        # for unmatched SKUs, DI offtake = 0
        df_all.quantity.fillna(0, inplace=True)
        # calcuate eib offtake
        df_all['EIB'] = df_all['volume'] - df_all['quantity']

        df_all = df_all[['date', 'sku_code', 'volume', 'quantity', 'EIB']]
        df_all = df_all.rename(columns={'volume': 'IL', 'quantity': "DI"})

        df = pd.melt(df_all, id_vars=['sku_code', 'date'], value_vars=['DI', 'IL', 'EIB'],
                     var_name='scope', value_name='volume')
        # create new columns
        df['type'] = 'offtake'
        df['unit'] = 'ton'
        df['status'] = 'actual'

        df = df.loc[df.date <= pd.to_datetime(today), :]  # keep actuals only

        # if str(os.name) == 'nt':  # keep leading zeros
        #     df['date'] = df['date'].dt.strftime('%m/%d/%Y')
        #     df['produced_date'] = today.strftime('%m/%d/%Y')
        # else:
        # Keep leading zero
        df['date'] = df['date'].dt.strftime('%m/%d/%Y')
        df['produced_date'] = today.strftime('%m/%d/%Y')

        # output
        df = df.loc[:, ['sku_code', 'scope', 'type', 'date', 'produced_date', 'status', 'volume', 'unit']]
        self.output_file(df, file_eib)

    def prep_apo_mappings(self) -> None:
        for entry in {'apo_channel_nfa_rfa', 'apo_stage_sku_split_7851', 'apo_stage_sku_split_7871'}:
            src_path = self.cfg[entry][F_SOURCE_PATH]
            basename = self.cfg[entry][F_BASENAME]
            filename, filepath = self.find_latest_file(
                file=entry,
                filename_wo_timestamp=basename,
                src_path=src_path
            )
            self.safe_copy_file(
                src_filepath=filepath,
                dest_filepath=os.path.join(DIR_MAPPINGS, self.cfg[entry][F_OUTPUT])
            )

    def prep_category_output_tmp(self) -> None:
        entry = 'category_output_tmp'
        src_path = self.cfg[entry][F_SOURCE_PATH]
        basename = self.cfg[entry][F_BASENAME]
        filename, filepath = self.find_latest_file(
            file=entry,
            filename_wo_timestamp=basename,
            src_path=src_path
        )
        self.safe_copy_file(
            src_filepath=filepath,
            dest_filepath=os.path.join(DIR_CACHE, self.cfg[entry][F_OUTPUT])
        )

    def prep_tin_to_ton(self,
                        file_il='di_tin_to_ton_mapping',
                        file_dc='productlist',
                        bcg_sku_mapping='bcg_sku_mapping'):
        """
        The purpose of this function is to prepare the tin-to-ton conversion table for IL
        """
        # load input
        df_tin2ton = self.load_latest_file(file_il)  # DI tin-to-ton mapping file
        df_sku = self.load_file(bcg_sku_mapping)  # sku code mapping file
        df_dc = self.load_latest_file(file_dc)

        # select relevant columns only for IL
        df_tin2ton = df_tin2ton.loc[:, ['SKU', 'Weight (gram) after downsize']].drop_duplicates()
        df_sku = df_sku.loc[:, ['SKU', 'SKU_wo_pkg']].drop_duplicates()

        # merge by SKU to get AF standard IL SKU code
        df = pd.merge(df_tin2ton, df_sku, how='left', on='SKU')
        df.drop('SKU', axis=1, inplace=True)
        df = df.loc[pd.notnull(df['SKU_wo_pkg']), :]

        # prepare DC mapping
        df_dc = df_dc.loc[:, ['SKU', 'weight_per_tin']]
        df_dc = df_dc.loc[pd.notnull(df_dc['SKU']), :]
        df_dc = df_dc.loc[pd.notnull(df_dc['weight_per_tin']), :]
        df_dc = df_dc.drop_duplicates()
        df_dc.rename(columns={'SKU': 'sku', 'weight_per_tin': 'format_in_grm'}, inplace=True)

        # rename columns to fit interface contract
        df.rename(columns={'SKU_wo_pkg': 'sku',
                           'Weight (gram) after downsize': 'format_in_grm'},
                  inplace=True)

        # order column to fit interface contract
        df = pd.concat([df, df_dc], sort=False)
        df = df.loc[:, ['sku', 'format_in_grm']]

        # export table
        self.output_file(df, file_il)

    def prep_uplift(self, file='il_uplift'):
        # load input file
        df = self.load_latest_file(file)

        # melt date columnes
        df = pd.melt(df, id_vars=['country', 'channel'], var_name='date', value_name='uplift')

        # convert uplift format from string to numeric (not necessary in latest versions)
        # df['uplift'] = df['uplift'].str.rstrip('%')
        # df['uplift'] = df['uplift'].astype(float) / 100

        # reorder columns
        df = df.loc[:, ['date', 'country', 'channel', 'uplift']]

        # export output
        self.output_file(df, file)

    def prep_og(self, file='il_og'):
        # load input file
        df = self.load_latest_file(file)

        # melt date columns
        df = pd.melt(df, id_vars=['country'], var_name='date', value_name='og')

        # reorder columns
        df = df.loc[:, ['date', 'country', 'og']]

        # export output
        self.output_file(df, file)

    def prep_unified_dc(self, file='unified_dc'):
        # load files
        df_dc = self.load_latest_file(file)
        df_dc_channel_split = pd.read_csv(os.path.join(DIR_CACHE, 'dc_channel_split.csv'))

        # melt date columns
        df_dc = pd.melt(df_dc, id_vars=['SKU (Tin)', 'Type'], value_name='volume', var_name=F_DN_MEA_DAT)

        # map type names to Impala code
        df_dc['Type'] = df_dc['Type'].replace({'ELN POS': F_DN_OFT_VAL,
                                               'sell out': F_DN_SAL_OUT_VAL,
                                               'sell in': F_DN_SAL_INS_VAL,
                                               'retailor inv.': F_DN_RTL_IVT_VAL,
                                               'retailor inv. Cvrg': F_DN_RTL_IVT_COV_VAL,
                                               'dis inv.': F_DN_SUP_IVT_VAL,
                                               'dis inv. Coverage': F_DN_SUP_IVT_COV_VAL})

        # force volume to be float and replace null
        df_dc['volume'] = df_dc['volume'].str.replace(',', '')
        df_dc['volume'] = df_dc['volume'].astype(float)
        df_dc['volume'].fillna(0, inplace=True)

        # pivot table
        df_dc = df_dc.pivot_table(index=['SKU (Tin)', F_DN_MEA_DAT], columns='Type', values='volume')
        df_dc = df_dc.reset_index().fillna("null")

        # rename columns
        df_dc.rename(columns={'SKU (Tin)': F_DN_MAT_COD}, inplace=True)

        # drop SKUs that are already in AF
        df_dc = df_dc.loc[~df_dc[F_DN_MAT_COD].isin(SELECTED_SKUS_DC), :]

        # prepare channel split table
        df_dc_channel_split.rename(columns={'sku_wo_pkg': F_DN_MAT_COD,
                                            'channel': F_DN_DIS_CHL_COD}, inplace=True)

        # merge DC data with channel split table
        df_dc_chl = pd.merge(df_dc, df_dc_channel_split, how='right', on=F_DN_MAT_COD)

        # apply split ratio and calculate volume per channel
        cols_val_to_split = [F_DN_OFT_VAL, F_DN_SAL_OUT_VAL, F_DN_SAL_INS_VAL, F_DN_RTL_IVT_VAL, F_DN_SUP_IVT_VAL]
        for col in cols_val_to_split:
            df_dc_chl[col] = df_dc_chl[col] * df_dc_chl['split_ratio']

        # clean outout
        df_dc_chl.drop(columns=['split_ratio'], inplace=True)
        df_dc_chl = df_dc_chl.loc[pd.notnull(df_dc_chl[F_DN_MEA_DAT]), :]

        # add additional fields
        df_dc_chl[F_DN_CYC_DAT] = datetime.date.today()
        df_dc_chl[F_DN_CRY_COD] = V_CN
        df_dc_chl[F_DN_LV2_UMB_BRD_COD] = V_DC
        df_dc_chl[F_DN_LV3_PDT_BRD_COD] = V_DC
        df_dc_chl[F_DN_LV4_PDT_FAM_COD] = df_dc_chl[F_DN_MAT_COD].str[:2]
        df_dc_chl[F_DN_LV5_PDT_SFM_COD] = df_dc_chl[F_DN_MAT_COD].str[:2]
        df_dc_chl[F_DN_LV6_PDT_NAT_COD] = df_dc_chl[F_DN_MAT_COD].str[2:3]
        df_dc_chl[F_DN_PCK_SKU_COD] = df_dc_chl[F_DN_MAT_COD]
        df_dc_chl[F_DN_FRC_USR_NAM_DSC] = ''
        df_dc_chl[F_DN_FRC_CRE_DAT] = datetime.date.today()
        df_dc_chl[F_DN_FRC_MDF_DAT] = datetime.date.today()
        df_dc_chl[F_DN_FRC_MTH_NBR] = 0
        df_dc_chl[F_DN_MEA_DAT] = pd.to_datetime(df_dc_chl[F_DN_MEA_DAT], format='%y-%b')
        df_dc_chl[F_DN_FRC_FLG] = df_dc_chl[F_DN_MEA_DAT] > pd.to_datetime(df_dc_chl[F_DN_CYC_DAT])
        df_dc_chl[F_DN_USR_NTE_TXT] = ''
        df_dc_chl[F_DN_APO_FLG] = False
        df_dc_chl[F_DN_OFT_TRK_VAL] = df_dc_chl[F_DN_OFT_VAL]

        df_dc_chl = df_dc_chl.loc[:, [F_DN_CYC_DAT,
                                      F_DN_CRY_COD,
                                      F_DN_LV2_UMB_BRD_COD,
                                      F_DN_LV3_PDT_BRD_COD,
                                      F_DN_LV4_PDT_FAM_COD,
                                      F_DN_LV5_PDT_SFM_COD,
                                      F_DN_LV6_PDT_NAT_COD,
                                      F_DN_MAT_COD,
                                      F_DN_PCK_SKU_COD,
                                      F_DN_DIS_CHL_COD,
                                      F_DN_FRC_USR_NAM_DSC,
                                      F_DN_FRC_CRE_DAT,
                                      F_DN_FRC_MDF_DAT,
                                      F_DN_FRC_MTH_NBR,
                                      F_DN_MEA_DAT,
                                      F_DN_FRC_FLG,
                                      F_DN_USR_NTE_TXT,
                                      F_DN_APO_FLG,
                                      F_DN_OFT_TRK_VAL,
                                      F_DN_OFT_VAL,
                                      F_DN_SAL_INS_VAL,
                                      F_DN_SAL_OUT_VAL,
                                      F_DN_RTL_IVT_VAL,
                                      F_DN_RTL_IVT_COV_VAL,
                                      F_DN_SUP_IVT_VAL,
                                      F_DN_SUP_IVT_COV_VAL]]

        # export output
        self.output_file(df_dc_chl, file)

    def prep_unified_di(self,
                        file_u1='unified_di_u1',
                        file_yuou='unified_di_yuou',
                        file_mapping='bcg_sku_mapping'):
        # load input
        df_u1 = self.load_latest_file(file_u1, skiprows=21)
        df_yuou = self.load_latest_file(file_yuou, skiprows=21)
        df_sku_mapping = self.load_file(file_mapping)

        # identify index columns for u1
        index_total = df_u1.columns.get_loc('*****')
        index_ali = df_u1.columns.get_loc('*****.1')
        index_bbc = df_u1.columns.get_loc('*****.2')
        range_total = np.r_[0:2, index_total + 1: index_ali - 2]
        range_ali = np.r_[0:2, index_ali + 1: index_bbc - 2]
        range_bbc = np.r_[0:2, index_bbc + 1: len(df_u1.columns) - 1]

        # clean up table
        df_u1 = pd.DataFrame(df_u1.values[1:], columns=df_u1.iloc[0])
        df_u1 = df_u1.loc[pd.notnull(df_u1['in tin']), :]
        df_u1 = df_u1.loc[df_u1['in tin'] != 'CBE - Total', :]

        # separate tables into three (by column index)
        # total
        df_u1_total = df_u1.iloc[:, range_total]
        cols_non_fy = [c for c in df_u1_total.columns if 'FY' not in c]
        df_u1_total = df_u1_total.loc[:, cols_non_fy]
        df_u1_total = pd.melt(df_u1_total, id_vars=['in tin', 'Type'], var_name='Date')
        df_u1_total['Channel'] = 'Total'

        # Ali
        df_u1_ali = df_u1.iloc[:, range_ali]
        cols_non_fy = [c for c in df_u1_ali.columns if 'FY' not in c]
        df_u1_ali = df_u1_ali.loc[:, cols_non_fy]
        df_u1_ali = pd.melt(df_u1_ali, id_vars=['in tin', 'Type'], var_name='Date')
        df_u1_ali['Channel'] = 'Ali'

        # BBC
        df_u1_bbc = df_u1.iloc[:, range_bbc]
        cols_non_fy = [c for c in df_u1_bbc.columns if 'FY' not in c]
        df_u1_bbc = df_u1_bbc.loc[:, cols_non_fy]
        df_u1_bbc = pd.melt(df_u1_bbc, id_vars=['in tin', 'Type'], var_name='Date')
        df_u1_bbc['Channel'] = 'BBC'

        # concatenate all sources
        df_u1 = pd.concat([df_u1_total, df_u1_ali, df_u1_bbc], sort=False)

        # identify index columns for yuou
        index_total = df_yuou.columns.get_loc('*****')
        index_new = df_yuou.columns.get_loc('*****.1')
        index_dist = df_yuou.columns.get_loc('*****.2')
        range_total = np.r_[0:2, index_total + 1: index_new - 2]
        range_new = np.r_[0:2, index_new + 1: index_dist - 2]
        range_dist = np.r_[0:2, index_dist + 1: len(df_yuou.columns) - 1]

        # clean up table
        df_yuou = pd.DataFrame(df_yuou.values[1:], columns=df_yuou.iloc[0])
        df_yuou = df_yuou.loc[pd.notnull(df_yuou['in tin']), :]
        df_yuou = df_yuou.loc[df_yuou['in tin'] != 'CBE - Total', :]

        # separate tables into three (by column index)
        # total
        df_yuou_total = df_yuou.iloc[:, range_total]
        cols_non_fy = [c for c in df_yuou_total.columns if 'FY' not in c]
        df_yuou_total = df_yuou_total.loc[:, cols_non_fy]
        df_yuou_total = pd.melt(df_yuou_total, id_vars=['in tin', 'Type'], var_name='Date')
        df_yuou_total['Channel'] = 'Total'

        # NewChannel
        df_yuou_nc = df_yuou.iloc[:, range_new]
        cols_non_fy = [c for c in df_yuou_nc.columns if 'FY' not in str(c)]
        df_yuou_nc = df_yuou_nc.loc[:, cols_non_fy]
        df_yuou_nc = pd.melt(df_yuou_nc, id_vars=['in tin', 'Type'], var_name='Date')
        df_yuou_nc['Channel'] = 'NewChannel'

        # Distribution
        df_yuou_dist = df_yuou.iloc[:, range_dist]
        cols_non_fy = [c for c in df_yuou_dist.columns if 'FY' not in str(c)]
        df_yuou_dist = df_yuou_dist.loc[:, cols_non_fy]
        df_yuou_dist = pd.melt(df_yuou_dist, id_vars=['in tin', 'Type'], var_name='Date')
        df_yuou_dist['Channel'] = 'Distribution'

        # concatenate all sources
        df_yuou = pd.concat([df_yuou_total, df_yuou_nc, df_yuou_dist], sort=False)

        # concatenate sp
        df_di = pd.concat([df_u1, df_yuou], sort=False)

        # format value column
        df_di['value'] = df_di['value'].str.replace(',', '')
        df_di['value'] = df_di['value'].str.replace(' - ', '0')
        df_di['value'] = df_di['value'].str.replace('#DIV/0!', '0')
        df_di['value'] = df_di['value'].astype(float)
        df_di['value'].fillna(0, inplace=True)

        # format date column
        df_di['Date'] = df_di['Date'].astype(int)
        df_di['Date'] = pd.to_datetime(df_di['Date'], format='%Y%m')

        # map type names
        df_di = df_di.loc[df_di['Type'] != 'Sellable stock', :]
        df_di['Type'] = df_di['Type'].replace({'Offtake': F_DN_OFT_VAL,
                                               'Sell out': F_DN_SAL_OUT_VAL,
                                               'Sell in (ETA)': F_DN_SAL_INS_VAL,
                                               'Retailer inv': F_DN_RTL_IVT_VAL,
                                               'Retailer inv.-month': F_DN_RTL_IVT_COV_VAL,
                                               'SP inv.-Total': F_DN_SUP_IVT_VAL,
                                               'SP inv.-month': F_DN_SUP_IVT_COV_VAL})

        # remove retailer inventory at SP level
        df_di_sp = df_di.loc[df_di['Channel'] != 'Total', :]
        df_di_sp = df_di_sp.loc[~df_di_sp['Type'].isin([F_DN_RTL_IVT_VAL, F_DN_RTL_IVT_COV_VAL]),
                   :]  # remove retailer inventory
        df_di_total = df_di.loc[df_di['Channel'] == 'Total', :]
        df_di = pd.concat([df_di_total, df_di_sp], sort=False)

        # map standardized sku
        df_sku_mapping = df_sku_mapping.loc[df_sku_mapping['Is_Mature'] == False, :]
        df_sku_mapping = df_sku_mapping.loc[:,
                         ['SKU', 'Country_acc', 'Brand_acc', 'Tier_acc', 'Stage_acc', 'Package_acc', 'SKU_wo_pkg']]

        df_di = pd.merge(df_di, df_sku_mapping, how='left', left_on='in tin', right_on='SKU')
        df_di = df_di.loc[pd.notnull(df_di['SKU']), :]  # remove unrecognized SKUs

        df_di.rename(columns={'Date': F_DN_MEA_DAT,
                              'Channel': F_DN_DIS_CHL_COD,
                              'Country_acc': F_DN_CRY_COD,
                              'Brand_acc': F_DN_LV4_PDT_FAM_COD,
                              'Tier_acc': F_DN_LV5_PDT_SFM_COD,
                              'Stage_acc': F_DN_LV6_PDT_NAT_COD,
                              'Package_acc': F_DN_PCK_SKU_COD,
                              'SKU_wo_pkg': F_DN_MAT_COD
                              }, inplace=True)

        # pivot table
        col_attribute = [F_DN_MEA_DAT, F_DN_DIS_CHL_COD, F_DN_CRY_COD, F_DN_LV4_PDT_FAM_COD, F_DN_LV5_PDT_SFM_COD,
                         F_DN_LV6_PDT_NAT_COD, F_DN_PCK_SKU_COD, F_DN_MAT_COD]
        df_di = df_di.loc[:, col_attribute + ['Type', 'value']]
        df_di = df_di.groupby(col_attribute + ['Type']).sum().reset_index()
        df_di = df_di.pivot_table(index=col_attribute, columns='Type', values='value')
        df_di = df_di.reset_index().fillna(0)

        # add required fields
        df_di[F_DN_CYC_DAT] = datetime.date.today()
        df_di[F_DN_LV2_UMB_BRD_COD] = V_IL
        df_di[F_DN_LV3_PDT_BRD_COD] = V_DI
        df_di[F_DN_FRC_USR_NAM_DSC] = ''
        df_di[F_DN_FRC_CRE_DAT] = datetime.date.today()
        df_di[F_DN_FRC_MDF_DAT] = datetime.date.today()
        df_di[F_DN_FRC_MTH_NBR] = 0
        df_di[F_DN_FRC_FLG] = df_di[F_DN_MEA_DAT] > pd.to_datetime(df_di[F_DN_CYC_DAT])
        df_di[F_DN_USR_NTE_TXT] = ''
        df_di[F_DN_APO_FLG] = False
        df_di[F_DN_OFT_TRK_VAL] = df_di[F_DN_OFT_VAL]

        # reorder columns
        df_di = df_di.loc[:, [F_DN_CYC_DAT,
                              F_DN_CRY_COD,
                              F_DN_LV2_UMB_BRD_COD,
                              F_DN_LV3_PDT_BRD_COD,
                              F_DN_LV4_PDT_FAM_COD,
                              F_DN_LV5_PDT_SFM_COD,
                              F_DN_LV6_PDT_NAT_COD,
                              F_DN_MAT_COD,
                              F_DN_PCK_SKU_COD,
                              F_DN_DIS_CHL_COD,
                              F_DN_FRC_USR_NAM_DSC,
                              F_DN_FRC_CRE_DAT,
                              F_DN_FRC_MDF_DAT,
                              F_DN_FRC_MTH_NBR,
                              F_DN_MEA_DAT,
                              F_DN_FRC_FLG,
                              F_DN_USR_NTE_TXT,
                              F_DN_APO_FLG,
                              F_DN_OFT_TRK_VAL,
                              F_DN_OFT_VAL,
                              F_DN_SAL_INS_VAL,
                              F_DN_SAL_OUT_VAL,
                              F_DN_RTL_IVT_VAL,
                              F_DN_RTL_IVT_COV_VAL,
                              F_DN_SUP_IVT_VAL,
                              F_DN_SUP_IVT_COV_VAL]]

        # export output
        self.output_file(df_di, file_u1)

    def prep_unified_il(self,
                        file_il_all='nonmature_il_all',
                        file_il_channel='nonmature_il_channel',
                        file_il_sellin='nonmature_il_sellin',
                        file_sku_mapping='nonmature_il_sku_mapping',
                        file_uplift='il_uplift'):
        # load files
        df_il_all = self.load_latest_file(file_il_all)
        df_il_chl = self.load_latest_file(file_il_channel)
        df_il_sellin = self.load_latest_file(file_il_sellin)
        df_uplift = self.load_latest_file(file_uplift)
        df_mapping = self.load_file(file_sku_mapping)

        # prepare IL total offtake

        # identify nonmature SKUs
        idx_nonmature = df_il_all[df_il_all.iloc[:, 0] == '*****'].index.values.astype(int)[0]
        df_il_all = pd.DataFrame(df_il_all.values[idx_nonmature + 2:], columns=df_il_all.iloc[idx_nonmature + 1])
        # rename first column (in case it changes)
        df_il_all.rename(columns={df_il_all.columns[0]: "brand"}, inplace=True)
        # drop excessive columns with no header
        df_il_all = df_il_all[df_il_all.columns.dropna()]
        # drop rows with no brand information (i.e. blank rows)
        df_il_all = df_il_all.loc[pd.notnull(df_il_all['brand']), :]
        # keep volume only
        df_il_all = df_il_all.loc[df_il_all['Trend Horizon'] == 'Current FCST', :]
        df_il_all.drop('Trend Horizon', axis=1, inplace=True)
        # exclude aggregate columns
        cols_valid = [c for c in df_il_all.columns if (('FY' not in c) and ('GR' not in c) and ('YTD' not in c))]
        df_il_all = df_il_all.loc[:, cols_valid]
        # create correct brand/SKU name
        df_il_all['sku'] = np.where(df_il_all['brand'].str.contains('Stage'), np.nan, df_il_all['brand'])
        df_il_all['sku'] = df_il_all['sku'].fillna(method='ffill')
        df_il_all = df_il_all.loc[df_il_all['brand'] != df_il_all['sku'], :]  # keep stage-level entries only
        df_il_all['stage'] = df_il_all['brand'].str[6:]
        df_il_all['sku_stage'] = df_il_all['sku'] + ' ' + df_il_all['stage']
        df_il_all.drop('brand', axis=1, inplace=True)
        # melt table (date columns)
        df_il_all = pd.melt(df_il_all, id_vars=['sku', 'stage', 'sku_stage'], value_name='offtake', var_name='date')
        # add attributes
        df_il_all = pd.merge(df_il_all, df_mapping, how='left', on='sku_stage')
        # format date
        df_il_all['date'] = pd.to_datetime(df_il_all['date'], format='%y-%b')

        # prepare sellin table
        df_il_sellin = df_il_sellin.iloc[:, :np.argmax(['Unnamed' in col for col in df_il_sellin.columns[1:]]) + 1]
        df_il_sellin = df_il_sellin.loc[pd.notnull(df_il_sellin['Description']), :]  # remove irrelevant content
        df_il_sellin = pd.melt(df_il_sellin, id_vars='Description', value_name='sellin', var_name='date')  # melt table
        df_il_sellin = df_il_sellin.loc[df_il_sellin['Description'].isin(df_mapping['sellin description'].unique()), :]
        df_il_sellin['date'] = pd.to_datetime(df_il_sellin['date'], format='%y-%b')  # reformat date
        df_il_sellin['sellin'] = df_il_sellin['sellin'].fillna(0)
        df_il_sellin = df_il_sellin.groupby(['Description', 'date']).sum().reset_index()  # prevent duplication
        df_il_sellin.rename(columns={'Description': 'sellin description'}, inplace=True)
        # merge with sellin
        df_il_all = pd.merge(df_il_all, df_il_sellin, how='left', on=['sellin description', 'date'])
        df_il_all['sellin'] = df_il_all['sellin'].fillna(0)

        # prepare channel split file

        # identify nonmature SKUs (new portfolio)
        idx_nonmature_chl = df_il_chl[df_il_chl.iloc[:, 0] == 'New Portfolio '].index.values.astype(int)[0]
        df_il_chl = pd.DataFrame(df_il_chl.values[idx_nonmature_chl + 2:],
                                 columns=df_il_chl.iloc[idx_nonmature_chl + 1])
        df_il_chl.rename(columns={df_il_chl.columns[0]: "sku_chl"},
                         inplace=True)  # rename first column (in case it changes)
        list_chl_col = df_il_chl.columns.tolist()
        # find first column with null header
        idx_null = [i for i, x in enumerate(list_chl_col) if pd.isnull(x)][0]
        df_il_chl = df_il_chl.iloc[:, 0:idx_null]  # keep columns until first null
        # exclude aggregate columns
        cols_valid = [c for c in df_il_chl.columns if (('FY' not in c)
                                                       and ('vs' not in c)
                                                       and ('Q1' not in c)
                                                       and ('Q2' not in c)
                                                       and ('Q3' not in c)
                                                       and ('Q4' not in c))]
        df_il_chl = df_il_chl.loc[:, cols_valid]
        # clean sku_chl column
        df_il_chl['sku_chl'] = df_il_chl['sku_chl'].replace({'AT - PF': 'AT PF',
                                                             'NL Pre-fea': 'NL Prefea',
                                                             'BBC/B2C': 'BChannel',
                                                             'C2C': 'CChannel',
                                                             'Offline store': 'Offline',
                                                             'Offline Store': 'Offline',
                                                             'offline store': 'Offline',
                                                             'WeChat Sales': 'WeChat',
                                                             'Oversea websites': 'OSW',
                                                             'Overseas websites': 'OSW',
                                                             'offline store&PDD': 'Offline'})
        list_channels = ['BChannel',
                         'CChannel',
                         'Offline',
                         'WeChat',
                         'OSW']

        list_skus_valid = df_il_all['sku'].unique().tolist()
        df_il_chl_valid = pd.DataFrame()
        df_il_chl = df_il_chl.reset_index(drop=True)  # reset index
        for s in list_skus_valid:
            if s not in df_il_chl['sku_chl'].tolist():
                print('No channel split found for SKU ' + s)
                continue
            idx_sku = df_il_chl[df_il_chl['sku_chl'] == s].index.values.astype(int)[0]
            df_temp = pd.DataFrame(df_il_chl.values[idx_sku:idx_sku + 9], columns=df_il_chl.columns)
            df_temp = df_temp.loc[df_temp['sku_chl'].isin(list_channels), :]  # keep rows with valid channels
            list_dates = df_temp.columns.tolist()
            list_dates.remove('sku_chl')
            for col in list_dates:
                df_temp[col] = df_temp[col].str.replace(',', '')
                df_temp[col] = df_temp[col].str.replace('-', '0')
                df_temp[col] = df_temp[col].astype(float)
                df_temp[col] = df_temp[col].fillna(0)
                num_channel_sum = df_temp[col].sum()
                if num_channel_sum == 0:
                    df_temp[col] = 1 / len(df_temp[col])
                else:
                    df_temp[col] = df_temp[col] / num_channel_sum
            df_temp['sku'] = s
            df_temp.rename(columns={'sku_chl': 'channel'}, inplace=True)
            df_il_chl_valid = pd.concat([df_il_chl_valid, df_temp], sort=False)
            del df_temp

        # melt table (date columns)
        df_il_chl_valid = pd.melt(df_il_chl_valid, id_vars=['sku', 'channel'], value_name='split_ratio',
                                  var_name='date')
        # convert date format
        df_il_chl_valid['date'] = pd.to_datetime(df_il_chl_valid['date'], format='%y-%b')

        # merge channel total with split ratio
        df_merged = pd.merge(df_il_all, df_il_chl_valid, how='left', on=['sku', 'date'])
        # apply split
        df_merged['offtake'] = df_merged['offtake'].astype(float) * df_merged['split_ratio']
        df_merged['sellin'] = df_merged['sellin'].astype(float) * df_merged['split_ratio']
        df_merged.drop('split_ratio', axis=1, inplace=True)

        # add uplift
        df_uplift = pd.melt(df_uplift, id_vars=['country', 'channel'], value_name='uplift', var_name='date')
        df_uplift['date'] = pd.to_datetime(df_uplift['date'])
        df_merged = pd.merge(df_merged, df_uplift, how='left', on=['country', 'channel', 'date'])
        df_merged['uplift'] = df_merged['uplift'].fillna(0)
        df_merged['uplift'] = df_merged['uplift'] + 1
        df_merged['total_offtake'] = df_merged['offtake'] * df_merged['uplift']

        # clean up columns
        df_merged = df_merged.loc[:,
                    ['SKU_WO_PKG', 'date', 'country', 'brand', 'tier', 'Stage', 'package', 'channel', 'offtake',
                     'sellin', 'total_offtake']]
        df_merged.rename(columns={'SKU_WO_PKG': F_DN_MAT_COD,
                                  'date': F_DN_MEA_DAT,
                                  'country': F_DN_CRY_COD,
                                  'brand': F_DN_LV4_PDT_FAM_COD,
                                  'tier': F_DN_LV5_PDT_SFM_COD,
                                  'Stage': F_DN_LV6_PDT_NAT_COD,
                                  'package': F_DN_PCK_SKU_COD,
                                  'channel': F_DN_DIS_CHL_COD,
                                  'offtake': F_DN_OFT_TRK_VAL,
                                  'sellin': F_DN_SAL_INS_VAL,
                                  'total_offtake': F_DN_OFT_VAL
                                  }, inplace=True)

        # add required fields
        df_merged[F_DN_SAL_OUT_VAL] = 0
        df_merged[F_DN_RTL_IVT_VAL] = 0
        df_merged[F_DN_RTL_IVT_COV_VAL] = 0
        df_merged[F_DN_SUP_IVT_VAL] = 0
        df_merged[F_DN_SUP_IVT_COV_VAL] = 0
        df_merged[F_DN_CYC_DAT] = datetime.date.today()
        df_merged[F_DN_LV2_UMB_BRD_COD] = V_IL
        df_merged[F_DN_LV3_PDT_BRD_COD] = V_IL
        df_merged[F_DN_FRC_USR_NAM_DSC] = ''
        df_merged[F_DN_FRC_CRE_DAT] = datetime.date.today()
        df_merged[F_DN_FRC_MDF_DAT] = datetime.date.today()
        df_merged[F_DN_FRC_MTH_NBR] = 0
        df_merged[F_DN_FRC_FLG] = df_merged[F_DN_MEA_DAT] > pd.to_datetime(df_merged[F_DN_CYC_DAT])
        df_merged[F_DN_USR_NTE_TXT] = ''
        df_merged[F_DN_APO_FLG] = False

        # reorder columns
        df_merged = df_merged.loc[:, [F_DN_CYC_DAT,
                                      F_DN_CRY_COD,
                                      F_DN_LV2_UMB_BRD_COD,
                                      F_DN_LV3_PDT_BRD_COD,
                                      F_DN_LV4_PDT_FAM_COD,
                                      F_DN_LV5_PDT_SFM_COD,
                                      F_DN_LV6_PDT_NAT_COD,
                                      F_DN_MAT_COD,
                                      F_DN_PCK_SKU_COD,
                                      F_DN_DIS_CHL_COD,
                                      F_DN_FRC_USR_NAM_DSC,
                                      F_DN_FRC_CRE_DAT,
                                      F_DN_FRC_MDF_DAT,
                                      F_DN_FRC_MTH_NBR,
                                      F_DN_MEA_DAT,
                                      F_DN_FRC_FLG,
                                      F_DN_USR_NTE_TXT,
                                      F_DN_APO_FLG,
                                      F_DN_OFT_TRK_VAL,
                                      F_DN_OFT_VAL,
                                      F_DN_SAL_INS_VAL,
                                      F_DN_SAL_OUT_VAL,
                                      F_DN_RTL_IVT_VAL,
                                      F_DN_RTL_IVT_COV_VAL,
                                      F_DN_SUP_IVT_VAL,
                                      F_DN_SUP_IVT_COV_VAL]]

        # export output
        self.output_file(df_merged, file_il_all)

    # Utils
    def load_file(self, file: str, key='input', encoding=None, usecols=None, skiprows=None,
                  header: Union[None, int] = 0,
                  skip_blank_lines=False) -> pd.DataFrame:
        """
        The purpose of this method is to read in csv.files from specified path and return a pandas df
        """

        if self.cfg[file]['source_path'] == '<mappings>':
            src_path = DIR_MAPPINGS
            copy_flag = False
        else:
            src_path = self.cfg[file]['source_path']
            copy_flag = True

        file_name = os.path.join(
            src_path,
            self.cfg[file][key]
        )

        df = pd.read_csv(file_name, encoding=encoding, usecols=usecols, skiprows=skiprows, header=header,
                         skip_blank_lines=skip_blank_lines)

        if copy_flag:
            self.safe_copy_file(
                src_filepath=file_name,
                dest_filepath=os.path.join(self.cfg[file][F_COPY_PATH], self.cfg[file][key])
            )

        if df.empty:
            raise ValueError(f'Input data ({file_name}) could not be loaded')

        return df

    def load_latest_file(self, file: str, key='input', encoding=None, usecols=None, skiprows=None,
                         header: Union[None, int] = 0, skip_blank_lines=False) -> pd.DataFrame:

        src_path = self.cfg[file][F_SOURCE_PATH]
        filename_wo_timestamp = self.cfg[file][key]
        filename, filepath = self.find_latest_file(file, filename_wo_timestamp, src_path)

        df = pd.read_csv(filepath, encoding=encoding, usecols=usecols, skiprows=skiprows, header=header,
                         skip_blank_lines=skip_blank_lines)

        self.safe_copy_file(
            src_filepath=filepath,
            dest_filepath=os.path.join(self.cfg[file][F_COPY_PATH], filename)
        )

        if df.empty:
            raise ValueError(f'Input data ({filepath}) could not be loaded')

        return df

    @staticmethod
    def find_latest_file(file, filename_wo_timestamp, src_path):
        # if file == 'di_sku_mapping':
        #     file_entries = [
        #         entry for entry in os.scandir(src_path)
        #         if entry.name == '20190612493609_CN3_AF_DI_SKU_Mapping.csv'
        #     ]
        #
        # else:
        #     file_entries = [
        #         entry for entry in os.scandir(src_path)
        #         if os.path.isfile(entry.path) and entry.name.endswith(filename_wo_timestamp)
        #     ]
        file_entries = [
            entry for entry in os.scandir(src_path)
            if os.path.isfile(entry.path) and entry.name.endswith(filename_wo_timestamp)
        ]
        if len(file_entries) == 0:
            raise FileNotFoundError(f'Expecting file like YYYYmmddHHMMSS{filename_wo_timestamp}. File not found.')

        def parse_entry_timestamp(entry_name) -> Union[None, datetime, int]:
            try:
                # return datetime.strptime(entry_name.split('_')[0], '%Y%m%d%H%M%S')
                return int(entry_name.split('_')[0])
            except Exception as e:
                logger.warning(f"""Value '{entry_name}' could not be parsed as datetime.\n{str(e)}""")
                raise ValueError(f"""Value '{entry_name}' could not be parsed as datetime.\n{str(e)}""")

        files_df = pd.DataFrame({
            F_FILENAME: [entry.name for entry in file_entries],
            F_FILE_TIMESTAMP: [parse_entry_timestamp(entry_name=entry.name) for entry in file_entries],
            F_FILE_ID: file,
            F_FILEPATH: [entry.path for entry in file_entries],
        })

        recent_entries_selector = files_df.groupby(by=F_FILE_ID)[F_FILE_TIMESTAMP].max().reset_index()
        selected_file: pd.DataFrame = pd.merge(
            left=files_df,
            right=recent_entries_selector,
            how='inner'
        )
        filepath = selected_file[F_FILEPATH][0]
        filename = selected_file[F_FILENAME][0]
        return filename, filepath

    @staticmethod
    def safe_copy_file(src_filepath: str, dest_filepath: str) -> None:
        try:
            shutil.copy2(src=src_filepath, dst=dest_filepath)
        except Exception as e:
            logger.warning(
                f'File {src_filepath} failed to copy to {dest_filepath}\n{str(e)}'
            )

    def output_file(self, df: pd.DataFrame, file_entry: str, encoding: str = 'utf_8') -> None:
        """
        The purpose of this method is to output interface contract format files to the specified path
        :param file_entry:
        :param encoding:
        :param df:
        :return:None
        """

        # write
        df.to_csv(
            os.path.join(DIR_DATA, self.cfg[file_entry][F_OUTPUT]),
            index=False,
            encoding=encoding,
            sep=';'
        )
        hdfs_filename = self.copy_to_hdfs(src_filename=self.cfg[file_entry][F_OUTPUT])
        self.create_quality_check(df_to_check=df, hdfs_filename=hdfs_filename)

    def business_contract_checks(self, bc_df: pd.DataFrame, file: str) -> None:
        """
        The purpose of this method is to check if the input file follows the business contracts
        """

        # checks
        # check if the number of columns is correct
        if len(list(bc_df)) != len(self.cfg[file]['business_contract'].keys()):
            raise ValueError('Number of columns incorrect, check business contract: ' + self.cfg[file]['input'])

        # check if the column names/fields are consistent with the business contract
        if set(list(bc_df)) != set(self.cfg[file]['business_contract'].keys()):
            raise ValueError('Column names inconsistent, check business contract: ' + self.cfg[file]['input'])

        # check if the datatype for all the columns are correct
        # if df.dtypes.to_dict() != self.cfg[file]['business_contract']:
        #     raise ValueError('Wrong column data types, check business contract: ' + self.cfg[file]['input'])

    # def export_mapping_file(self, df: pd.DataFrame, file: str):
    #     """
    #     The purpose of this method is to export cleaned mapping files for later use
    #     """
    #     df.to_csv(
    #         os.path.join(
    #             self.data_path,
    #             self.cfg[file]['output']
    #         ),
    #         index=False)
    #     self.output_file(df=df, file_entry=file)

    @staticmethod
    def sku_name_to_scope_brand_country_tier_source_cols(df: pd.DataFrame, sku_name_col: str) -> pd.DataFrame:
        """
        The purpose of this method is to convert a column with string sku names to multiple columns indicating scope/
        brand, country and tier
        :param df:
        :param sku_name_col:
        :return:pandas dataframe
        """

        def sku_name_to_scope(row):
            full_name = row[sku_name_col]
            scope = ''
            if "International" in full_name:
                scope = 'IL'
            elif "China" in full_name:
                scope = 'DC'
            elif "OIB" in full_name:
                scope = 'DC'
            return scope

        def sku_name_to_country(row):
            full_name = row[sku_name_col]
            country = 'Total'
            if "ANZ" in full_name:
                country = "ANZ"
            elif "Germany" in full_name:
                country = "DE"
            elif "German" in full_name:
                country = "DE"
            elif "Platinium" in full_name:
                country = "NL"
            elif "UK" in full_name:
                country = "UK"
            elif "HK" in full_name:
                country = "HK"
            elif "others" in full_name:
                country = "others"
            return country

        def sku_name_to_tier(row):
            full_name = row[sku_name_col]
            tier = ''
            list_tiers_1 = ['PF', 'PN']

            for item in list_tiers_1:
                if item in full_name:
                    tier = item

            if 'Gold' in full_name:
                tier = 'GD'
            elif 'goat' in full_name:
                tier = 'GT'
            elif 'Cow & Gate' in full_name:
                tier = 'C&G'
            elif 'Profutura' in full_name:
                tier = 'PF'
            return tier

        # standardize brand names
        def sku_name_to_brand(row):
            """
            The purpose of this
            :param row:
            :return:
            """
            full_name = row['Brand']
            brand = full_name
            list_brands = ['Aptamil', 'Karicare', 'Cow & Gate', 'Nutrilon', 'Nutricia',
                           'Meadjohnson', 'Friso', 'Wyeth', 'Nestle', 'MEIJI', 'Abbott']

            for item in list_brands:
                if item in full_name:
                    brand = item

            return brand

        df['Scope'] = df.apply(sku_name_to_scope, axis=1)
        df['Country'] = df.apply(sku_name_to_country, axis=1)
        df['Tier'] = df.apply(sku_name_to_tier, axis=1)
        df['Brand'] = df.apply(sku_name_to_brand, axis=1)
        df['Geography'] = ''
        df['Source'] = 'Smartpath'  # identify the data source

        return df

    @staticmethod
    def remove_comma_in_numeric_columns(df: pd.DataFrame, whitelist: str) -> pd.DataFrame:
        """
        The purpose of this function is to remove all the comma in numeric columns
        :param df:
        :param whitelist: list of non-numeric columns
        :return: df with numeric columns without comma
        """
        columns = df.columns
        for col in columns:
            if col != whitelist:
                df.loc[:, col] = df.loc[:, col].str.replace(',', '').astype(float)
        return df

    def copy_to_hdfs(self, src_filename) -> str:
        filename_split = src_filename.split('.')
        filename_with_timestamp = '.'.join(filename_split[:-1]) \
                                  + datetime.now().strftime('_%Y%m%d.') \
                                  + filename_split[-1]
        # commented by ZX, to avoid overwrite on files in HDFS folders, notice '-f' should be added to force overwrite
        # (ret, out, err) = run_cmd(['hdfs', 'dfs', '-put','-f',
        #                            os.path.join(DIR_DATA, src_filename),
        #                            os.path.join(self.cfg['paths']['hdfs_dest'], filename_with_timestamp)])
        # logger.debug(ret)
        # logger.debug(out)
        # logger.debug(err)
        return filename_with_timestamp

    # backup (obsolete)
    def prep_dc_sap_sellin(self, file='sap_sellin'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # transform
        df.rename(columns={'BillT': 'order_type',
                           'Billing Date': 'date',
                           'Sold-To Pt': 'SP_code',
                           'Material': 'SKU_NO',
                           'Billed Quantity': 'quantity',
                           'SU': 'unit'},
                  inplace=True)
        df = df.loc[df['date'] >= '2019-01-01', :]  # do not update historical data (due to legacy definition)
        df = df[df['SOrg.'] == 7850]  # code 7850 stands for DC
        # list_sales_type = ['ZF2', 'ZCI']
        list_return_type = ['ZRE', 'ZS1']  # return orders to be deducted
        df['adj'] = np.where(df['order_type'].isin(list_return_type),
                             df['quantity'] * (-1),
                             df['quantity'])
        df.drop('quantity', axis=1, inplace=True)
        df.drop('Bill.Doc.', axis=1, inplace=True)
        df.drop('SOrg.', axis=1, inplace=True)
        df['type'] = 'sellin'
        df['scope'] = 'DC'
        df.rename(columns={'adj': 'quantity'}, inplace=True)

        # output file
        df = df.loc[:, ['date', 'order_type', 'SP_code', 'SKU_NO', 'scope', 'type', 'quantity', 'unit']]
        self.output_file(df, file)

    def prep_category_input(self, file='category_input'):
        # load file
        df = self.load_file(file)

        # check (not needed as BC = IC)
        # self.business_contract_checks(df, file)

        # output
        self.output_file(df, file)

    # def prep_legacy_sku_code_mapping(self, file='old_SKU_code_mapping'):
    #     # load file
    #     df = self.load_file(file)
    #
    #     # check
    #     self.business_contract_checks(df, file)
    #
    #     # transform
    #     df['Old SAP Code'] = df['Old SAP Code'].str.strip()
    #     df['Abbreviation'] = df['Abbreviation'].str.strip()
    #     df.rename(columns={'Old SAP Code': 'SKU_NO',
    #                        'Abbreviation': 'SKU',
    #                        'THEMIS SAP Material Number': 'SKU_NO_new'},
    #               inplace=True)
    #
    #     # output
    #     self.export_mapping_file(df, file)

    def create_quality_check(self, df_to_check: pd.DataFrame, hdfs_filename: str) -> None:
        qc_columns = [
            'rec_typ_dsc', 'obj_typ_dsc', 'fil_nam_dsc', 'fil_col_nbr', 'fil_lin_nbr',
            'col_nam_dsc',
            'col_min_val', 'col_max_val',
            'col_avg_val', 'col_std_dev_val', 'col_dis_val_dsc', 'col_dis_val_nbr',
            'col_ept_nbr',
            't_rec_ins_tst',
        ]
        qc_num_cols = len(df_to_check.columns)
        qc_num_rows = len(df_to_check.index)

        qc_df = pd.DataFrame(columns=qc_columns)
        for col in df_to_check.columns:
            col_sf = df_to_check[col]

            if is_numeric_dtype(col_sf):
                to_add = {
                    'col_nam_dsc': col,
                    'col_min_val': col_sf.min(skipna=True),
                    'col_max_val': col_sf.max(skipna=True),
                    'col_avg_val': col_sf.mean(skipna=True),
                    'col_std_dev_val': col_sf.std(skipna=True),
                    'col_dis_val_dsc': '',
                    'col_dis_val_nbr': len(col_sf.drop_duplicates(inplace=False).index),
                    'col_ept_nbr': col_sf.isna().sum(),
                    'rec_typ_dsc': V_QUANTITATIVE,
                    't_rec_ins_tst': self.tst
                }

            else:
                distinct_values_str = col_sf.drop_duplicates(inplace=False).apply(lambda x: str(x)).str.cat(sep=', ')
                to_add = {
                    'col_nam_dsc': col,
                    'col_min_val': 0,
                    'col_max_val': 0,
                    'col_avg_val': 0,
                    'col_std_dev_val': 0,
                    'col_dis_val_dsc': distinct_values_str[:500],
                    'col_dis_val_nbr': len(col_sf.drop_duplicates(inplace=False).index),
                    'col_ept_nbr': col_sf.isna().sum(),
                    'rec_typ_dsc': V_QUALITATIVE,
                    't_rec_ins_tst': self.tst
                }

            qc_df = qc_df.append(to_add, ignore_index=True)

        qc_df['env_dsc'] = ''
        qc_df['cbu_cod'] = ''
        qc_df['cbu_dsc'] = ''
        qc_df['mod_dsc'] = ''
        qc_df['err_typ_cod'] = ''
        qc_df['err_typ_dsc'] = ''
        qc_df['obj_typ_dsc'] = 'File'
        qc_df['fil_nam_dsc'] = hdfs_filename
        qc_df['fil_col_nbr'] = qc_num_cols
        qc_df['fil_lin_nbr'] = qc_num_rows

        qc_df.fillna(value=0, inplace=True)

        with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
            impala_cfg = yaml.load(ymlf)
        impala_connection = ImpalaUtils.get_impala_connector(cfg=impala_cfg[F_CONNECT])
        impala_cursor = impala_connection.cursor()
        sql_insert_req = SQLUtils.make_insert_req_from_dataframe_infer_types(
            to_database=impala_cfg[F_QUALITY_CHECKS][F_DATABASE],
            to_table=impala_cfg[F_QUALITY_CHECKS][F_TABLE],
            df=qc_df
        )
        impala_cursor.execute(sql_insert_req)
        impala_connection.close()
