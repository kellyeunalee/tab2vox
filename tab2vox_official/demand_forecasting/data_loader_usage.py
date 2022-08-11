
import pandas as pd
import numpy as np
import itertools 
import os
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from demand_forecasting.multicolumn_label_encoder import MultiColumnLabelEncoder
from demand_forecasting.custom_dataset import MND_Dataset


def split_xy_from_data(file_path: str, 
                       dummy_cols: list,
                       key_features: list,
                       label_name: str):
 
    data = pd.read_csv(file_path).drop(dummy_cols, axis=1)

    key_x_features = [x for x in data.columns.to_list() if x not in [label_name]]
    key_y_features = key_features + [label_name]

    key_x = data[key_x_features]
    key_y = data[key_y_features]
    
    df_key_x = key_x[data['ATTRI_YR'].str.contains('2019') == False]    
    df_key_y = key_y[data['ATTRI_YR'].str.contains('2019') == True]     

    return df_key_x, df_key_y

        
class Voxel:
    
    def __init__(self, key_features, df_train, df_valid, df_test): 
        
        self.all_x = pd.concat([df_train, df_valid, df_test], axis=0) 
           
        self.bases = self.all_x.BASE_BIGO.unique()
        self.equipments = self.all_x.EQUIP_NIIN.unique()
        self.years = self.all_x.ATTRI_YR.unique()       
        
        self.cnt_bases = len(self.bases)                
        self.cnt_equipments = len(self.equipments)       
        self.cnt_years = len(self.years)                
            
        self.train_items = df_train.ITM_NIIN.unique()   
        self.valid_items = df_valid.ITM_NIIN.unique()   
        self.test_items = df_test.ITM_NIIN.unique()     
       
        self.key_features = key_features                
        
    def reshape_to_4d(self, items):   

        unique_keys = pd.DataFrame(itertools.product(items, 
                                                     self.bases, 
                                                     self.equipments, 
                                                     self.years), 
                                   columns=self.key_features)

        df_block = pd.merge(unique_keys, self.all_x,     
                            left_on = self.key_features, right_on = self.key_features, 
                            how = 'left').fillna(0)  

        tensor_block = torch.from_numpy(df_block.values).float()

        tensor_4d_block = tensor_block.reshape(len(items), 
                                               self.cnt_bases, 
                                               self.cnt_equipments, 
                                               self.cnt_years, 
                                               -1)                                      
        return tensor_4d_block
    
    def make_4d_tensor(self):
        
        tensor_4d_train_x = self.reshape_to_4d(self.train_items)
        tensor_4d_valid_x = self.reshape_to_4d(self.valid_items)
        tensor_4d_test_x = self.reshape_to_4d(self.test_items)
        
        return tensor_4d_train_x, tensor_4d_valid_x, tensor_4d_test_x

 
def get_loaders(config):
    if config.dataset == 'MND':
        train_loader, valid_loader, test_loader = MND_get_loaders(config)
    return train_loader, valid_loader, test_loader

 
def MND_get_loaders(config):

    dataset_train = 'data/A01_DATA_MND_SET1_TRAIN.csv'
    dataset_test = 'data/A01_DATA_MND_SET1_TEST.csv'

    dummy_cols = ['SET1', 'SET2', 'SET3', 'SET4', 'SET5', 'SEED', 'RAND_NO', 'TRAIN_CNT', 'TEST_CNT']
    key_features = ['ITM_NIIN', 'BASE_BIGO', 'EQUIP_NIIN', 'ATTRI_YR']   

    train_x, train_y = split_xy_from_data(dataset_train, dummy_cols, key_features, 'LABEL') 
    test_x, test_y  = split_xy_from_data(dataset_test, dummy_cols, key_features, 'LABEL')     

    unscl_train_x, unscl_train_y, unscl_test_x, unscl_test_y = train_x.copy(), train_y.copy(), test_x.copy(), test_y.copy()

    train_x["ITM_NIIN"].nunique()     
    
    item_list = train_x['ITM_NIIN'].unique()

    if config.model == 'tab2vox':   
        train_item, valid_item = train_test_split(item_list, train_size=config.tab2vox_train_ratio, random_state=123) 

    df_train_key_x = train_x.set_index('ITM_NIIN').loc[train_item].reset_index()    
    df_valid_key_x = train_x.set_index('ITM_NIIN').loc[valid_item].reset_index()    
    df_train_key_y = train_y.set_index('ITM_NIIN').loc[train_item].reset_index()    
    df_valid_key_y = train_y.set_index('ITM_NIIN').loc[valid_item].reset_index()    
    
    numeric_cols = [x for x in df_train_key_x.columns.to_list() if x not in key_features]   
    base_cols = df_train_key_x['BASE_BIGO'].unique()     
    equip_cols = df_train_key_x['EQUIP_NIIN'].unique()   

    scaler = StandardScaler()
    scaler.fit(df_train_key_x.loc[:, numeric_cols])

    df_train_key_x.loc[:, numeric_cols] = scaler.transform(df_train_key_x.loc[:, numeric_cols])
    df_valid_key_x.loc[:, numeric_cols] = scaler.transform(df_valid_key_x.loc[:, numeric_cols])
    test_x.loc[:, numeric_cols] = scaler.transform(test_x.loc[:, numeric_cols])
    
    x_key_features = pd.concat([train_x[key_features], test_x[key_features]], axis=0) 
    y_key_features = pd.concat([train_y[key_features], test_y[key_features]], axis=0) 
    all_key_features = pd.concat([x_key_features, y_key_features], axis=0)
    
    mce = MultiColumnLabelEncoder(columns = key_features).fit(all_key_features)

    df_train_key_x.loc[:, key_features] = mce.transform(df_train_key_x.loc[:, key_features])
    df_valid_key_x.loc[:, key_features] = mce.transform(df_valid_key_x.loc[:, key_features])
    df_train_key_y.loc[:, key_features] = mce.transform(df_train_key_y.loc[:, key_features])
    df_valid_key_y.loc[:, key_features] = mce.transform(df_valid_key_y.loc[:, key_features])

    test_x.loc[:, key_features] = mce.transform(test_x.loc[:, key_features])    
    test_y_mce_trans = mce.transform(test_y.loc[:, key_features])
    test_y_all = pd.concat([test_y, test_y_mce_trans], axis = 1)

    if config.model == 'tab2vox':

        x_feature_cluster = torch.load('./results/feature_clustering/clustered_x_feature_grp5')
        bases_cluster = torch.load('./results/feature_clustering/clustered_bases_grp3')
        equip_cluster = torch.load('./results/feature_clustering/clustered_equips_grp2')
        
        voxel = Voxel(key_features, df_train_key_x, df_valid_key_x, test_x)
        
        tensor_4d_train_x, tensor_4d_valid_x, tensor_4d_test_x = voxel.make_4d_tensor()

        train_sum_y = df_train_key_y.groupby('ITM_NIIN')['LABEL'].sum()  
        valid_sum_y = df_valid_key_y.groupby('ITM_NIIN')['LABEL'].sum()  
        test_sum_y = test_y.groupby('ITM_NIIN')['LABEL'].sum()           

        order_train_ITM_NIIN = df_train_key_y['ITM_NIIN'].unique()
        order_valid_ITM_NIIN = df_valid_key_y['ITM_NIIN'].unique()
        order_test_ITM_NIIN = test_y['ITM_NIIN'].unique()
        
        train_sum_y = train_sum_y.reindex(order_train_ITM_NIIN)
        valid_sum_y = valid_sum_y.reindex(order_valid_ITM_NIIN)
        test_sum_y = test_sum_y.reindex(order_test_ITM_NIIN)

        train_loader = DataLoader(  
            dataset=MND_Dataset(
                data=tensor_4d_train_x, 
                labels=torch.from_numpy(train_sum_y.values).float(),
                key_features=None,
                algo_type='voxel'
                ),  
            batch_size = config.tab2vox_batch_size,
            shuffle=True,)  
        
        valid_loader = DataLoader(
            dataset=MND_Dataset(
                data=tensor_4d_valid_x, 
                labels=torch.from_numpy(valid_sum_y.values).float(), 
                key_features=None,
                algo_type='voxel'
                ),
            batch_size = config.tab2vox_batch_size,
            shuffle=True,)
                    
        test_loader = DataLoader(
            dataset=MND_Dataset(
                data=tensor_4d_test_x, 
                labels=torch.from_numpy(test_sum_y.values).float(), 
                key_features=None,
                algo_type='voxel'
                ),
            batch_size = 1, 
            shuffle=False,)
        
        torch.save({'train_loader': train_loader,
                    'valid_loader': valid_loader,
                    'test_loader': test_loader,
                    }, './data/tab2vox_loader_test1_bs2')
                         
        return train_loader, valid_loader, test_loader 

 