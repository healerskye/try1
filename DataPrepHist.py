# coding: utf-8
import datetime
import logging
import os
from datetime import date
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


F_FILE_ID = 'file_id'
F_FILE_TIMESTAMP = 'file_timestamp'


class DataPrepHist:

    # Constructors
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def prep_smartpath(self, file_id: str = 'smartpath') -> None:
        smartpath_df = self.load_file(file=file_id)

        # Split the dataframe
        smartpath_hierarchy_dict = self.cfg[file_id]['hierarchy']

        smartpath_df_pseudo_index = smartpath_df.loc[:, 'Import analysis'].apply(lambda x: str(x).strip())
        smartpath_df = smartpath_df.loc[:,
                       self.cfg[file_id]['timeaxis']['columns']['start']:self.cfg[file_id]['timeaxis']['columns']['end']
                       ]
        smartpath_df.columns = smartpath_df.loc[
                               smartpath_df_pseudo_index == self.cfg[file_id]['timeaxis']['row'],
                               self.cfg[file_id]['timeaxis']['columns']['start']:
                               self.cfg[file_id]['timeaxis']['columns']['end']
                               ].iloc[0].tolist()
        smartpath_df['Import analysis'] = smartpath_df_pseudo_index

        def read_chunk(url_stack: list) -> pd.DataFrame:
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
        list_of_sp_value_dict = [dict for dict in dataframes if dict['url'][0] == "Value (RMB '000)"]
        list_of_sp_volume_dict = [dict for dict in dataframes if dict['url'][0] == "Volume (Ton)"]

        def initial_transformation(dict, value_name: str) -> pd.DataFrame:
            """
            The purpose of this function is to conduct some initial transformation of the data while reading in
            the dictionaries from the list of dictionaries
            :param dict: this is the dictionary with dict['dataframe'] = the chunk of data needs to be read in
            :param value_name: indicates the val_col after unpivoting the data
            :return:
            """
            # read in df
            df = dict['dataframe']
            # change column name for the brand info
            df = df.rename(columns={'Import analysis': 'Brand'})
            # columns to unpivot
            value_vars = list(df)
            value_vars.remove('Brand')
            # unpivot df
            df_long = pd.melt(df, id_vars='Brand', value_vars=value_vars, var_name='date', value_name=value_name)
            # add channel info
            df_long['channel'] = dict['url'][-1]  # the last level in the hierarchical structure indicates the channel
            df_long = df_long[df_long['Brand'].notnull()]
            # reformat date
            df_long['date'] = pd.to_datetime([datetime.strptime(date, '%y-%b') for date in df_long['date']])

            return df_long

        def generate_df(list_of_dict: list, value_name: str) -> pd.DataFrame:
            """
            The purpose of this function is to read in the dictionaries in the list of volume/value dictionaries
            defined above
            :param list_of_dict: list of sp value/volume dictionaries
            :param value_name: indicates the val_col after unpivoting the data
            :return: df after initial transformations
            """
            df = pd.DataFrame()
            for dict in list_of_dict:  # dataframes is a list of dictionaries
                df_long = initial_transformation(dict, value_name)
                df = pd.concat([df, df_long])

            return df

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
        order_cols = ['Brand', 'Scope', 'Country', 'Geography', 'Source', 'channel', 'date', 'volume_ton', 'value_kRMB',
                      'price_kRMB_per_ton']
        sp = sp[order_cols]

        # step 8: for all records with volume = 0 and value = 0, set price = 0
        df = sp.copy()
        df['price_kRMB_per_ton'].fillna(0, inplace=True)

        # step 9: elect IL and selected channels
        list_channels = ['B2C', 'BBC', 'C2C']
        df_IL = df.query(
            '(Scope == "IL")&(Country == "Total")&(Source == "Smartpath")&(channel in @list_channels)').copy()

        # output file
        self.output_file(df_IL, file_id)

    def prep_osa_eib(self, file_id: str = 'osa_eib'):
        osa_eib_df = self.load_file(file=file_id)

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

        # Transform percentages into decimal values
        # osa_eib_df['value'] = osa_eib_df['value'].str[:-1].astype(float) / 100

        # output file
        self.output_file(osa_eib_df, file_id)

    def prep_price_eib(self, file_id: str = 'price_eib'):
        # Warning: one column in the data file contains a typo: 'Source counry'
        price_df = self.load_file(file=file_id)

        lookup_df = pd.read_csv(os.path.join(self.data_path, self.cfg[file_id]['brand_and_tier_mapping_sheet']),
                                encoding='ISO-8859-1')
        sku_mapping = pd.read_csv(os.path.join(self.data_path, self.cfg[file_id]['sku_mapping']), encoding='ISO-8859-1')

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
        # price_df = price_df.set_index('sku_code') # set first column as index in case index is not dropped when exported

        # output file
        self.output_file(price_df, file_id)

    def prep_sellin_eib(self, file_id: str = 'sellin_eib'):
        sellin_df = self.load_file(file=file_id, skiprows=1)

        lookup_df = pd.read_csv(os.path.join(self.data_path, self.cfg[file_id]['brand_and_tier_mapping_sheet']),
                                encoding='ISO-8859-1')
        sku_mapping = pd.read_csv(os.path.join(self.data_path, self.cfg[file_id]['sku_mapping']), encoding='ISO-8859-1')

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

        # date_list = sellin_df.columns[~pd.to_datetime(sellin_df.columns, errors='coerce').isnull()].values.tolist()
        date_list = [s for s in sellin_df.columns if s.isdigit()]

        sellin_df.rename({'SKU_wo_pkg': 'sku_code'}, axis=1, inplace=True)
        sellin_df = sellin_df.loc[:, date_list + ['sku_code', 'scope']]
        sellin_df = sellin_df.set_index(['sku_code', 'scope'])
        sellin_df = sellin_df.stack().reset_index().rename({'level_2': 'date', 0: 'value'}, axis=1)
        sellin_df['date'] = pd.TimedeltaIndex(sellin_df['date'].astype(int), unit='d') + datetime(1900, 1, 1)
        sellin_df['date'] = [date.replace(day=1) for date in sellin_df['date']]

        sellin_df['status'] = 'actual'
        sellin_df.loc[sellin_df.date.dt.to_period('M') > pd.to_datetime('today').to_period('M'),
                      'status'] = 'forecasted'

        sellin_df['type'] = 'Sell-in'
        sellin_df['unit'] = 'ton'
        sellin_df['produced_date'] = 'Default'

        sellin_df = sellin_df.loc[:, ['sku_code', 'scope', 'type', 'date', 'produced_date', 'status', 'value', 'unit']]
        # sellin_df = sellin_df.set_index('sku_code')

        # output file
        self.output_file(sellin_df, file_id)

    def prep_legacy_sku_code_mapping(self, file='old_SKU_code_mapping'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # transform
        df['Old SAP Code'] = df['Old SAP Code'].str.strip()
        df['Abbreviation'] = df['Abbreviation'].str.strip()
        df.rename(columns={'Old SAP Code': 'SKU_NO',
                           'Abbreviation': 'SKU',
                           'THEMIS SAP Material Number': 'SKU_NO_new'},
                  inplace=True)

        # output
        self.export_mapping_file(df, file)

    def prep_legacy_sp_code_mapping(self, file='old_sp_code_mapping'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # output
        self.export_mapping_file(df, file)

    def prep_dc_anp(self, file='AnP'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # transform
        df = pd.melt(df, id_vars=['Date'], value_vars=['AP', 'AC', 'NC', 'C&G', 'Karicare', 'Happy Family'])
        df.rename(columns={'variable': 'Brand', 'value': 'Spending'}, inplace=True)
        df['Date'] = pd.TimedeltaIndex(df['Date'], unit='d') + datetime(1900, 1, 1)

        # output
        self.output_file(df, file)

    def prep_dc_osa(self, file='DC_OSA'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # output
        self.output_file(df, file)

    def prep_dc_store_dist(self, file='dc_store_dist'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # output
        self.output_file(df, file)

    def prep_productlist(self, file='productlist'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # transformation
        df.rename(columns={'SKU_type': 'SKU_Type',
                           'product_name': 'Name',
                           'SKU': 'Stage',
                           'brand': 'Brand',
                           'weight_per_unit': 'UnitWeight',
                           'price': 'UnitPrice',
                           'unit': 'Unit',
                           'unit_per_case': 'CaseUnit'},
                  inplace=True)
        df['Brand'] = df['Brand'].replace({'白金诺优能': 'NP',
                                           '可瑞康': 'KG',
                                           '诺优能': 'NC',
                                           '多美滋': 'DG',
                                           '爱他美': 'AC',
                                           '牛栏': 'CG',
                                           '白金爱他美': 'AP'})

        # output file
        self.output_file(df, file)

    def prep_distributor(self, file='distributorlist'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # transform
        df.rename(columns={'Channel': 'Group'}, inplace=True)

        # output file
        self.output_file(df, file)

    def prep_customers(self, file='customerlist'):
        # load file
        df = self.load_file(file, encoding='utf_8')

        # check
        self.business_contract_checks(df, file)

        # transform: remove invalid stores
        df = df.loc[~df['store_name'].str.contains("QS虚拟门店"), :]

        # output file
        self.output_file(df, file, encoding='utf_8')

    def prep_pos(self, file='pos'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # transform
        # add new columns
        df['scope'] = 'DC'
        df['unit'] = 'tin'
        df['type'] = 'offtake'
        # clean date format
        df['date'] = df['date'].astype(str) + '15'  # we add a day number so that it is read as a date
        df['date'] = pd.to_datetime(df['date'])

        # output
        self.output_file(df, file)

    def prep_dms_sellout(self, file='dms_sellout'):
        # load file
        df = self.load_file(file)

        # check
        self.business_contract_checks(df, file)

        # transformation
        df.drop('SP_code', axis=1, inplace=True)
        df.drop('customer_code', axis=1, inplace=True)
        # add new columns
        df['scope'] = 'DC'
        df['unit'] = 'TIN'
        df['type'] = 'sellout'
        df = df.rename(columns={'SP_value': 'revenue'})
        df['price'] = df['revenue'] / df['quantity']

        # output file
        self.output_file(df, file)

    def prep_sp_inv(self, file='sp_inv'):
        # load file
        df = self.load_file(file)
        df_sku = pd.read_csv(os.path.join(
            self.data_path,
            self.cfg['old_SKU_code_mapping']['output']
        ))

        # check
        self.business_contract_checks(df, file)

        # transformation
        # drop null SKU_NO
        df.dropna(axis=0, subset=['SKU_NO'], inplace=True)
        # add SKU_NO by two levels of matching (old and new)
        df = pd.merge(df, df_sku[['SKU_NO', 'SKU']], on='SKU_NO', how='left')
        df_new_sku = df_sku[['SKU_NO_new', 'SKU']]
        df_new_sku['SKU_NO_new'] = df_new_sku['SKU_NO_new'].apply(str)
        df = pd.merge(df, df_new_sku, left_on='SKU_NO', right_on='SKU_NO_new', how='left')
        df['SKU_x'] = df['SKU_x'].fillna(df['SKU_y'])
        df.drop('SKU_y', axis=1, inplace=True)
        df.rename(columns={'SKU_x': 'SKU'}, inplace=True)
        # add new columns
        df['scope'] = 'DC'
        df['unit'] = 'Tin'
        df['type'] = 'sp_inv'
        df.drop('SKU_NO_new', axis=1, inplace=True)
        df = df.rename(columns={'SP_value': 'revenue'})
        df['price'] = df['revenue'] / df['quantity']

        # output file
        self.output_file(df, file)

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
        self.output_file(df, file)

    def prep_category_input(self, file='category_input'):
        # load file
        df = self.load_file(file)

        # check (not needed as BC = IC)
        # self.business_contract_checks(df, file)

        # output
        self.output_file(df, file)

    def prep_IL_sellin(self, file='il_sellin'):
        file_name_DI = os.path.join(
            self.output_path,
            self.cfg['di_tradeflow']['output']
        )
        file_name_EIB = os.path.join(
            self.output_path,
            self.cfg['sellin_eib']['output']
        )
        df_DI = pd.read_csv(file_name_DI)
        df_EIB = pd.read_csv(file_name_EIB)

        # prepare DI data
        df_DI = df_DI.loc[df_DI['type'] == 'sellin', :]
        df_DI = df_DI.loc[:, ['SKU_wo_pkg', 'status', 'date', 'quantity']]
        df_DI = df_DI.groupby(['SKU_wo_pkg', 'status', 'date']).sum().reset_index()
        df_DI.rename(columns={'quantity': 'DI', 'SKU_wo_pkg': 'sku_code'}, inplace=True)

        # prepare EIB data
        df_EIB = df_EIB.loc[:, ['sku_code', 'date', 'status', 'value']]
        df_EIB.rename(columns={'value': 'EIB'}, inplace=True)

        # merge DI and EIB data
        df_IL_sellin = pd.merge(df_DI, df_EIB, how='left', on=['date', 'sku_code', 'status'])
        df_IL_sellin = pd.melt(df_IL_sellin, id_vars=['date', 'sku_code', 'status'], value_vars=['DI', 'EIB'])
        df_IL_sellin['value'] = df_IL_sellin['value'].fillna(0)
        df_IL_sellin['type'] = 'Sell-in'
        df_IL_sellin['produced_date'] = 'Default'
        df_IL_sellin['unit'] = 'ton'

        self.output_file(df_IL_sellin, file)

    def prep_di_tradeflow(self, file='di_tradeflow', bcg_sku_mapping='bcg_sku_mapping',
                          tin_ton_mapping='di_tin_to_ton_mapping'):
        today = date.today()
        date_m_1 = datetime(today.year, today.month - 1, 1)
        date_m_4_plan = datetime(today.year, today.month + 4, 1)
        date_m_3_plan = datetime(today.year, today.month + 3, 1)

        def prep_di_u1_sellout(file='di_u1_sellout', mapping_file='di_customer_mapping_U1_sellout', sku_mapping=
        'di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df = pd.merge(df, df_mapping[['客户', 'channel']], on='客户', how='left')
            df = df.loc[df['状态'] == '已发货']
            df['date'] = pd.TimedeltaIndex(df['U1系统发货日期'], unit='d') + datetime(1900, 1, 1)
            df['date_for_matching'] = [date.replace(day=1) for date in df['date']]
            df = df.loc[df['date_for_matching'] == date_m_1]
            df['sp'] = 'U1'
            df['type'] = 'sellout'
            df['scope'] = 'DI'
            df['status'] = 'actual'
            df = pd.merge(df, df_sku_mapping[['U1 sellout/offtake', 'trade flow SKU desc']], left_on='SKU', right_on=
            'U1 sellout/offtake', how='left')
            select_cols = ["trade flow SKU desc", "sp", "type", "channel", "scope", "status", "date_for_matching", "销量"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "type", "channel", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            # output
            return df

        def prep_di_u1_offtake(file='di_u1_offtake', mapping_file='di_customer_mapping_U1_offtake', sku_mapping=
        'di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df = pd.merge(df, df_mapping[['Retailer', 'channel']], on='Retailer', how='left')
            df['date_for_matching'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
            df = df.loc[df['date_for_matching'] == date_m_1]
            df['sp'] = 'U1'
            df['type'] = 'offtake'
            df['scope'] = 'DI'
            df['status'] = 'actual'
            df = pd.merge(df, df_sku_mapping[['U1 sellout/offtake', 'trade flow SKU desc']], left_on='U1 SKU', right_on=
            'U1 sellout/offtake', how='left')

            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "volume in tin"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            return df

        def prep_di_yuou_sellout(file='di_yuou_sellout', mapping_file='di_customer_mapping_yuou_sellout', sku_mapping=
        'di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df = pd.merge(df, df_mapping[['客户', 'channel']], on='客户', how='left')
            df['date'] = pd.TimedeltaIndex(df['渝欧系统发货日期'], unit='d') + datetime(1900, 1, 1)
            df['date_for_matching'] = [date.replace(day=1) for date in df['date']]
            df = df.loc[pd.to_datetime(df['date_for_matching']) == date_m_1]
            df = df.loc[df['状态'] == '已发货']
            df['sp'] = 'Yuou'
            df['type'] = 'sellout'
            df['scope'] = 'DI'
            df['status'] = 'actual'
            df = pd.merge(df, df_sku_mapping[['Yuou sellout/offtake', 'trade flow SKU desc']], left_on='SKU', right_on=
            'Yuou sellout/offtake', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching", "销量"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            return df

        def prep_di_yuou_offtake(file='di_yuou_offtake', file_yunji='di_yuou_yunji_offtake',
                                 mapping_file='di_customer_mapping_yuou_sellout', sku_mapping=
                                 'di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_yunji = self.load_latest_file(file_yunji)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df['date'] = pd.TimedeltaIndex(df['渝欧系统发货日期'], unit='d') + datetime(1900, 1, 1)
            df['date_for_matching'] = [date.replace(day=1) for date in df['date']]
            df['date_for_matching'] = [date.replace(day=1) for date in df['date']]
            df = df.loc[df['date_for_matching'] == date_m_1]

            df_yunji['date'] = pd.TimedeltaIndex(df_yunji['渝欧系统发货日期'], unit='d') + datetime(1900, 1, 1)
            df_yunji['date_for_matching'] = [date.replace(day=1) for date in df_yunji['date']]
            df_yunji = df_yunji.loc[df_yunji['date_for_matching'] == date_m_1]

            df = df.loc[df['状态'] == '已发货']
            df_yunji = df_yunji.loc[df_yunji['状态'] == '已发货']

            df = df.loc[df['客户'] != 'YUNJI']
            df = pd.concat([df, df_yunji], ignore_index=True, sort=True)

            df = pd.merge(df, df_mapping[['客户', 'channel']], on='客户', how='left')

            df['sp'] = 'Yuou'
            df['type'] = 'offtake'
            df['scope'] = 'DI'
            df['status'] = 'actual'

            df = pd.merge(df, df_sku_mapping[['Yuou sellout/offtake', 'trade flow SKU desc']], left_on='SKU', right_on=
            'Yuou sellout/offtake', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching", "销量"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            return df

        def prep_di_u1_sp_inv(file='di_u1_sp_inventory', sku_mapping='di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df = df.loc[df['验收类型'] == '正品']
            df = df.loc[df['商户'] != 'pop']
            df['sp'] = 'U1'
            df['type'] = 'sp_inv'
            df['scope'] = 'DI'
            df['status'] = 'actual'
            df['channel'] = 'all'
            df['date'] = date_m_1
            df = pd.merge(df, df_sku_mapping[['U1 SP inv', 'trade flow SKU desc']], left_on='品牌SKU', right_on=
            'U1 SP inv', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date", "金蝶数量"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            return df

        def prep_di_yuou_sp_inv(file='di_yuou_sp_inventory', sku_mapping='di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df['date'] = date_m_1
            df = df.loc[df['商户名称'].isin(['待销毁', '待销毁冻结', '废品', '残次品', '临期冻结'])]
            df['sp'] = 'Yuou'
            df['type'] = 'sp_inv'
            df['scope'] = 'DI'
            df['status'] = 'actual'
            df['channel'] = 'all'
            df = pd.merge(df, df_sku_mapping[['Yuou SP inv', 'trade flow SKU desc']], left_on='达能SKU', right_on=
            'Yuou SP inv', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date", "数量"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            # output
            return df

        def prep_di_u1_retailer_inv(file='di_u1_retailer_inventory', mapping_file='di_customer_mapping_u1_retailer_inv',
                                    sku_mapping='di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_mapping = self.load_latest_file(mapping_file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df = df[np.isfinite(df['月末库存1'])]
            df['date'] = pd.to_datetime([datetime.strptime(str(int(date)), '%Y%m') for date in df['月末库存1']])
            df['sp'] = 'U1'
            df['type'] = 'retailer_inv'
            df['scope'] = 'DI'
            df['status'] = 'actual'
            df = pd.merge(df, df_mapping[['客户名称', 'channel']], on='客户名称', how='left')

            df = pd.merge(df, df_sku_mapping[['U1 retailer inv', 'trade flow SKU desc']], left_on='品牌-SKU', right_on=
            'U1 retailer inv', how='left')
            df = df.loc[df['date'] == date_m_1]
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date", "安全库存数量"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')
            return df

        def prep_di_yuou_retailer_inv(file='di_yuou_retailer_inventory', sku_mapping='di_sku_mapping'):
            # load file
            df = self.load_latest_file(file)
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transform
            df['date_for_matching'] = pd.to_datetime(
                [datetime.strptime(date, '%Y%m') for date in df['Month'].astype(str)])
            df = df.loc[df['date_for_matching'] == date_m_1]
            df['channel'] = 'all'
            df['sp'] = 'Yuou'
            df['type'] = 'retailer_inv'
            df['scope'] = 'DI'
            df['status'] = 'actual'
            df = pd.merge(df, df_sku_mapping[['Yuou retailer inv (to be designed)', 'trade flow SKU desc']],
                          left_on='SKU', right_on='Yuou retailer inv (to be designed)', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "Volume"]
            df = df[select_cols]
            # df.loc[:, 'Volume'] = df.loc[:, 'Volume'].str.replace(',', '').astype(float)
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')

            return df

        def prep_di_u1_sellin(file='di_u1_sellin', sku_mapping='di_sku_mapping'):
            # load file
            df = self.load_latest_file(file, encoding='unicode_escape')
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transformation
            df_dates_column_names = df.iloc[0, 5:].tolist()
            df_other_column_names = df.iloc[1, :5].tolist()
            df = df.iloc[2:, :]
            df.columns = df_other_column_names + df_dates_column_names
            na_filter_cols = ["Plant", "Brand", "Code", "Description"]
            df.dropna(subset=na_filter_cols, how='all', inplace=True)  # filter totals
            df = pd.melt(df, id_vars=df_other_column_names, value_vars=df_dates_column_names, var_name='date',
                         value_name='volume')
            df = df.loc[df['Status'] == 'TTL']
            df['date'] = pd.TimedeltaIndex(df['date'].astype(int), unit='d') + datetime(1900, 1, 1)
            df['date_for_matching'] = [date.replace(day=1) for date in df['date']]
            df = df[['Brand', 'Plant', 'date_for_matching', 'volume']]

            def define_status(row):
                if (row['Plant'] == 'Aintree') & (row['date_for_matching'] > date_m_1) & (
                        row['date_for_matching'] <= date_m_3_plan):
                    return 'forecasted'
                if (row['Plant'] != 'Aintree') & (row['date_for_matching'] > date_m_1) & (
                        row['date_for_matching'] <= date_m_4_plan):
                    return 'forecasted'
                if (row['date_for_matching'] == date_m_1):
                    return 'actual'
                return 'Other'

            df['status'] = df.apply(lambda row: define_status(row), axis=1)

            df = df.loc[df['status'].isin(['forecasted', 'actual'])]
            df['sp'] = 'U1'
            df['type'] = 'sellin'
            df['scope'] = 'DI'
            df['channel'] = 'all'

            df = pd.merge(df, df_sku_mapping[['sell in', 'trade flow SKU desc']], left_on='Brand', right_on=
            'sell in', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "volume"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df.loc[:, 'quantity'] = df.loc[:, 'quantity'].str.replace(',', '').astype(float)

            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg(
                'sum')

            return df

        def prep_di_yuou_sellin(file='di_yuou_sellin', sku_mapping='di_sku_mapping'):
            # load file
            df = self.load_latest_file(file, encoding='unicode_escape')
            df_sku_mapping = self.load_latest_file(sku_mapping)

            # transformation
            df_dates_column_names = df.iloc[0, 5:].tolist()
            df_other_column_names = df.iloc[1, :5].tolist()
            df = df.iloc[2:, :]
            df.columns = df_other_column_names + df_dates_column_names
            na_filter_cols = ["Plant", "Brand", "Code", "Description"]
            df.dropna(subset=na_filter_cols, how='all', inplace=True)  # filter totals
            df = pd.melt(df, id_vars=df_other_column_names, value_vars=df_dates_column_names, var_name='date',
                         value_name='volume')
            df = df.loc[df['Status'] == 'TTL']
            df['date'] = pd.TimedeltaIndex(df['date'].astype(int), unit='d') + datetime(1900, 1, 1)
            df['date_for_matching'] = [date.replace(day=1) for date in df['date']]
            df = df[['Brand', 'Plant', 'date_for_matching', 'volume']]

            def define_status(row):
                if (row['Plant'] == 'Aintree') & (row['date_for_matching'] > date_m_1) & (
                        row['date_for_matching'] <= date_m_3_plan):
                    return 'forecasted'
                if (row['Plant'] != 'Aintree') & (row['date_for_matching'] > date_m_1) & (
                        row['date_for_matching'] <= date_m_4_plan):
                    return 'forecasted'
                if (row['date_for_matching'] == date_m_1):
                    return 'actual'
                return 'Other'

            df['status'] = df.apply(lambda row: define_status(row), axis=1)

            df = df.loc[df['status'].isin(['forecasted', 'actual'])]
            df['sp'] = 'Yuou'
            df['type'] = 'sellin'
            df['scope'] = 'DI'
            df['channel'] = 'all'

            df = pd.merge(df, df_sku_mapping[['sell in', 'trade flow SKU desc']], left_on='Brand', right_on=
            'sell in', how='left')
            select_cols = ["trade flow SKU desc", "sp", "channel", "type", "scope", "status", "date_for_matching",
                           "volume"]
            df = df[select_cols]
            df.columns = ["sku", "sp", "channel", "type", "scope", "status", "date", "quantity"]
            df.loc[:, 'quantity'] = df.loc[:, 'quantity'].str.replace(',', '').astype(float)
            df = df.groupby(["sku", "sp", "type", "channel", "scope", "status", "date"], as_index=False)[
                'quantity'].agg('sum')

            return df

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

        # concatenate all di files
        df = pd.concat(
            [df_u1_offtake, df_u1_retailer_inv, df_u1_sellout, df_u1_sp_inv, df_yuou_offtake, df_yuou_sellout,
             df_yuou_sp_inv, df_u1_retailer_inv, df_yuou_retailer_inv, df_u1_sellin, df_yuou_sellin],
            ignore_index=True)

        # map di sku to BCG sku
        bcg_sku_mapping = self.load_file(bcg_sku_mapping)
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
        df['produced_date'] = today
        df['unit'] = 'ton'
        df = df[
            ['SKU_std', 'SKU_wo_pkg', 'sku', 'country', 'brand', 'tier', 'stage', 'package', 'sp', 'type', 'channel',
             'scope', 'status', 'date', 'produced_date', 'quantity_ton', 'unit']]
        df = df.rename(columns={'quantity_ton': 'quantity',
                                'SKU_std': 'sku_code'})

        self.output_file(df, file)

    def prep_il_offtake(self, file='il_offtake'):
        today = date.today()
        date_m_2 = datetime(today.year, today.month - 2, 1)

        # step 1: loading data at channel*brand*country level
        def il_automation_load_raw_data(data_path, row_mapping_path):
            il_offtake = self.load_latest_file(data_path, encoding='ISO-8859-1', skip_blank_lines=False, header=None)
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

        def il_automation_load_raw_data_osw(data_path, row_mapping_path):

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
            il_offtake['date'] = [date.replace(day=1) for date in il_offtake['date']]

            return il_offtake

        def combine_df(list_of_df: list):
            df = pd.concat(list_of_df)

            # rename brand to acc. names
            df.loc[df['brand'].str.contains('Aptamil'), 'brand'] = "APT"
            df.loc[df['brand'].str.contains('Karicare'), 'brand'] = "KC"
            df.loc[df['brand'].str.contains('Cow & Gate'), 'brand'] = "C&G"
            df.loc[df['brand'].str.contains('Nutrilon'), 'brand'] = "NC"

            # select data from month m_2
            # df = df.loc[df.date == date_m_2]

            return df

        # step 2: loading data at sku level
        def il_automation_load_wechat_sku(file):

            il_offtake = self.load_latest_file(file, encoding='utf-8', skip_blank_lines=True, header=None)

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
                [datetime.strptime(date, '%Y%m.0') for date in il_offtake['date'].astype(str)])

            il_offtake = il_offtake.rename(columns={'Description': 'cn description'})
            il_offtake = il_offtake[['cn description', 'date', 'volume']]
            il_offtake['channel'] = 'wechat'

            return il_offtake

        def il_automation_load_ec_sku(file):

            il_offtake = self.load_latest_file(file, encoding='utf-8', skip_blank_lines=True, header=None)

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
                [datetime.strptime(date, '%Y%m.0') for date in il_offtake['date'].astype(str)])

            il_offtake = il_offtake[['CN Description', 'date', 'volume']]
            il_offtake['channel'] = 'EC'
            il_offtake.columns = il_offtake.columns.str.lower()

            return il_offtake

        def combine_sku_data(list_of_df: list, mapping_data='il_sku_database'):

            # loading
            df_sku = pd.concat(list_of_df)
            df_sku_mapping = self.load_file(mapping_data)
            df_sku_mapping = df_sku_mapping.query('Group == "IL"')[['SKU', 'Country', 'Brand_acc', 'Description CN']]
            df_sku_mapping = df_sku_mapping.rename(columns={'Description CN': 'cn description', 'Brand_acc': 'brand'})
            df_sku_mapping.columns = df_sku_mapping.columns.str.lower()

            # rename CN description for mapping
            df_sku['cn description'] = df_sku['cn description'].replace({
                ## KC ANZ (Australia + New Zealand)
                '澳大利亚金装可瑞康1段': '金装可瑞康1段',
                '新西兰金装可瑞康1段': '金装可瑞康1段',
                '澳大利亚金装可瑞康2段': '金装可瑞康2段',
                '新西兰金装可瑞康2段': '金装可瑞康2段',
                '澳大利亚金装可瑞康3段': '金装可瑞康3段',
                '新西兰金装可瑞康3段': '金装可瑞康3段',
                '澳大利亚金装可瑞康4段': '金装可瑞康4段',
                '新西兰金装可瑞康4段': '金装可瑞康4段',
                ## AC DE
                '德国爱他美1+': '德国爱他美4段 (1岁以上)',
                '德国爱他美2+': '德国爱他美5段 (2岁以上)',
                '德国爱他美pre段': '德国爱他美PRE段',
                ## AC PF ANZ
                '澳洲爱他美白金版1段': '澳洲爱他美白金1段',
                '澳洲爱他美白金版2段': '澳洲爱他美白金2段',
                '澳洲爱他美白金版3段': '澳洲爱他美白金3段',
                '澳洲爱他美白金版4段': '澳洲爱他美白金4段',
                ## AC PN ANZ
                '澳洲爱他美金装1段': '澳洲金装爱他美1段',
                '澳洲爱他美金装2段': '澳洲金装爱他美2段',
                '澳洲爱他美金装3段': '澳洲金装爱他美3段',
                '澳洲爱他美金装4段': '澳洲金装爱他美4段',
                ## AC UK
                '英国爱他美 1段': '英国爱他美1段',
                '英国爱他美 2段': '英国爱他美2段',
                '英国爱他美 3段': '英国爱他美3段 (1岁以上)',
                '英国爱他美 4段': '英国爱他美4段 (2岁以上)',
                ## AC PF UK
                '英国爱他美白金1段': '英国爱他美白金版1段',
                '英国爱他美白金2段': '英国爱他美白金版2段',
                '英国爱他美白金3段': '英国爱他美白金版3段',
                ## C&G UK
                '英国牛栏3段': '英国牛栏3段 (1岁以上)',
                '英国牛栏4段': '英国牛栏4段 (2岁以上)'})

            # get tier and stage info from sku info
            def get_tier(df):
                SKU = df['sku']
                tier = ''
                list_tiers = ['PN', 'PF', 'COW', 'GOAT', 'C&G']
                for elem in list_tiers:
                    if elem in SKU:
                        tier = elem
                return tier

            def get_stage(df):
                SKU = df['sku']
                stage = ''
                list_stages = ['PRE', '1', '2', '3', '4', '5', '6']
                for elem in list_stages:
                    if elem in SKU:
                        stage = elem
                return stage

            def get_sku_split(df_sku, df_sku_mapping):
                df_sku = pd.merge(df_sku, df_sku_mapping, on='cn description')
                df_sku['tier'] = df_sku.apply(get_tier, axis=1)
                df_sku['stage'] = df_sku.apply(get_stage, axis=1)
                df_sku.columns = df_sku.columns.str.lower()
                df_sku['volume'] = pd.to_numeric(df_sku.volume, errors='coerce')
                return df_sku

            df_sku = get_sku_split(df_sku, df_sku_mapping)

            # select only last month data
            # df_sku = df_sku.loc[df_sku.date == date_m_2]

            return df_sku

        # step 3: get tier and stage level % from df_sku and apply them to all skus in df

        def cal_sku_volume(df, df_sku):
            volume_tier = df_sku.groupby(['country', 'brand', 'tier', 'stage', 'date'],
                                         as_index=False)['volume'].agg('sum')
            volume_brand = df_sku.groupby(['country', 'brand', 'date'], as_index=False)['volume'].agg('sum')
            df_ratio = pd.merge(volume_tier, volume_brand, on=['country', 'brand', 'date'],
                                suffixes=['_tier_stage', '_brand'])
            df_ratio['ratio_brand'] = df_ratio['volume_tier_stage'] / df_ratio['volume_brand']
            df = pd.merge(df, df_ratio[['country', 'brand', 'tier', 'stage', 'date', 'ratio_brand']])

            df['volume'] = df['value'].astype(float) * df['ratio_brand']
            df = df[['date', 'country', 'brand', 'tier', 'stage', 'channel', 'volume']]

            # rename selected tier names for standardization
            df.loc[df['tier'] == 'COW', 'tier'] = 'GD'
            df.loc[df['tier'] == 'GOAT', 'tier'] = 'GT'

            df['sku_code'] = df['country'] + '_' + df['brand'] + '_' + df['tier'] + '_' + df['stage']
            df['scope'] = 'IL'
            df['type'] = 'offtake'
            df['unit'] = 'ton'
            df['produced_date'] = today
            df['status'] = 'actual'

            df = df[
                ['sku_code', 'scope', 'country', 'brand', 'tier', 'stage', 'type', 'date', 'produced_date', 'status',
                 'volume', 'channel', 'unit']]

            return df

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

        # output files
        df = df[['sku_code', 'scope', 'type', 'date', 'produced_date', 'status', 'volume', 'channel', 'unit']]
        self.output_file(df, file)

    def prep_eib_offtake(self, file_di='di_tradeflow', file_il='il_offtake', file_eib='eib_offtake'):
        today = date.today()
        date_m_2 = datetime(today.year, today.month - 2, 1)
        # load files
        il_file_name = os.path.join(
            self.cfg[file_il]['transformed_path'],
            self.cfg[file_il]['output'])

        df_il_offtake = pd.read_csv(il_file_name)

        # di_file_name = os.path.join(
        #     self.cfg[file_di]['transformed_path'],
        #     self.cfg[file_di]['output'])

        di_file_name = os.path.join(
            self.cfg[file_eib]['source_path'],
            self.cfg[file_eib]['di_offtake_hist']
        )
        df_di_tradeflow = pd.read_csv(di_file_name)

        # check if the data is empty
        if df_il_offtake.empty:
            raise ValueError('Input data is empty')

        # transformation
        df_di_offtake = df_di_tradeflow.loc[df_di_tradeflow['type'] == 'offtake']
        df_di_offtake = df_di_offtake.loc[df_di_offtake['channel'] != 'Total']
        df_di_offtake_by_sku = df_di_offtake.groupby(['date', 'SKU_wo_pkg'], as_index=False)['quantity'].agg('sum')
        df_di_offtake_by_sku['date'] = pd.to_datetime(df_di_offtake_by_sku.date)
        df_di_offtake_by_sku = df_di_offtake_by_sku.loc[df_di_offtake_by_sku['date'] <= date_m_2]

        df_il_offtake_by_sku = df_il_offtake.groupby(['date', 'sku_code'], as_index=False)['volume'].agg('sum')
        df_il_offtake_by_sku['date'] = pd.to_datetime(df_il_offtake_by_sku.date)
        df_all = pd.merge(df_il_offtake_by_sku,
                          df_di_offtake_by_sku,
                          left_on=['sku_code', 'date'], right_on=['SKU_wo_pkg', 'date'],
                          how='left')

        df_all.quantity.fillna(0, inplace=True)  # for unmatched SKUs, DI offtake = 0

        df_all['EIB'] = df_all['volume'] - df_all['quantity']

        df_all = df_all[['date', 'sku_code', 'volume', 'quantity', 'EIB']]
        df_all = df_all.rename(columns={'volume': 'IL', 'quantity': "DI", 'sku_code': 'SKU_wo_pkg'})

        df = pd.melt(df_all, id_vars=['SKU_wo_pkg', 'date'], value_vars=['DI', 'IL', 'EIB'],
                     var_name='scope', value_name='volume')
        df['type'] = 'offtake'
        df['unit'] = 'ton'
        df['produced_date'] = today

        # output
        self.output_file(df, file_eib)

    # Utils
    def load_file(self, file: str, key='input', encoding=None, usecols=None, skiprows=None, header=0,
                  skip_blank_lines=False) -> pd.DataFrame:
        """
        The purpose of this method is to read in csv.files from specified path and return a pandas df
        """
        file_name = os.path.join(
            self.cfg[file]['source_path'],
            self.cfg[file][key]
        )

        df = pd.read_csv(file_name, encoding=encoding, usecols=usecols, skiprows=skiprows, header=header,
                         skip_blank_lines=skip_blank_lines)

        # check if the data is empty
        if df.empty:
            raise ValueError('Input data is empty')

        return df

    def load_latest_file(self, file: str, key='input', encoding=None, usecols=None, skiprows=None, header=0,
                         skip_blank_lines=False) -> pd.DataFrame:

        file_entries = [
            entry for entry in os.scandir(self.cfg[file]['source_path'])
            if os.path.isfile(entry.path) and entry.name.endswith(self.cfg[file][key])
        ]

        files_df = pd.DataFrame({
            F_FILENAME: [entry.name for entry in file_entries],
            F_FILE_TIMESTAMP: [datetime.strptime(entry.name.split('_')[0], '%Y%m%d%H%M%S') for entry in file_entries],
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

        df = pd.read_csv(filepath, encoding=encoding, usecols=usecols, skiprows=skiprows, header=header,
                         skip_blank_lines=skip_blank_lines)

        # check if the data is empty
        if df.empty:
            raise ValueError('Input data is empty')

        return df

    def output_file(self, df: pd.DataFrame, file: str, encoding: str = 'utf_8') -> None:
        """
        The purpose of this method is to output interface contract format files to the specified path
        :param file:
        :param encoding:
        :param df:
        :return:
        """

        # write
        df.to_csv(
            os.path.join(
                self.cfg[file]['transformed_path'],
                self.cfg[file]['output']
            ),
            index=False,
            encoding=encoding
        )

    def business_contract_checks(self, df: pd.DataFrame, file: str) -> None:
        """
        The purpose of this method is to check if the input file follows the business contracts
        :param df:
        :param file:
        :return:
        """

        # checks
        # check if the number of columns is correct
        if len(list(df)) != len(self.cfg[file]['business_contract'].keys()):
            raise ValueError('Number of columns incorrect, check business contract: ' + self.cfg[file]['input'])

        # check if the column names/fields are consistent with the business contract
        if set(list(df)) != set(self.cfg[file]['business_contract'].keys()):
            raise ValueError('Column names inconsistent, check business contract: ' + self.cfg[file]['input'])

        # check if the datatype for all the columns are correct
        # if df.dtypes.to_dict() != self.cfg[file]['business_contract']:
        #     raise ValueError('Wrong column data types, check business contract: ' + self.cfg[file]['input'])

    def export_mapping_file(self, df: pd.DataFrame, file: str):
        """
        The purpose of this method is to export cleaned mapping files for later use
        """
        df.to_csv(
            os.path.join(
                self.data_path,
                self.cfg[file]['output']
            ),
            index=False)
        self.output_file(df=df, file=file)

    def sku_name_to_scope_brand_country_tier_source_cols(self, df: pd.DataFrame, sku_name_col: str) -> pd.DataFrame:
        """
        The purpose of this method is to convert a column with string sku names to multiple columns indicating scope/
        brand, country and tier
        :param df:
        :param sku_name_col:
        :return:
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

    def remove_comma_in_numeric_columns(self, df: pd.DataFrame, whitelist: str) -> pd.DataFrame:
        """
        The purpose of this function is to remove all the comma in numeric columns
        :param df:
        :param whitelist: list of non-numeric columns
        :return: df with numeric columns without comma
        """
        columns = df.columns
        for col in columns:
            if col != whitelist:
                df[col] = df[col].str.strip()
                df[col] = df[col].replace('', 0)
                df[col].fillna(0, inplace=True)
                df.loc[:, col] = df.loc[:, col].astype(str).str.replace(',', '').astype(float)
        return df
