
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import torch

df_train_x = torch.load('./data/df_train_key_x')

key_features = df_train_x.columns.tolist()[0:4]     

base_features = df_train_x['BASE_BIGO'].unique()
equip_features = df_train_x['EQUIP_NIIN'].unique()
numeric_features = [x for x in df_train_x.columns.tolist() if x not in key_features]  

def hierachical_clustering(df_train_x):
    equips_cluster = equips_clustering(df_train_x)
    clustered_equips = np.vstack((equip_features, equips_cluster)).T
    torch.save(clustered_equips, './results/feature_clustering/clustered_equips_grp3')

def bases_clustering(df_train_x):
    pivoted_train_x = pd.pivot_table(
        df_train_x,
        values = numeric_features,
        index = 'BASE_BIGO',
        columns = ['ITM_NIIN', 'EQUIP_NIIN', 'ATTRI_YR'],
        )
    pivoted_train_x.reset_index(inplace=True)
    
    pivoted_train_x.dropna(how='all', axis=1, inplace=True) 

    col_mean = pivoted_train_x.mean()
    pivoted_train_x.fillna(col_mean, inplace = True) 

    flatten_columns = [str(x) for x, y, z, w in pivoted_train_x.columns if y == '' or z == '' or w == ''] + [str(x) + '_' + str(y) + '_' + str(z) + '_' + str(w) for x, y, z, w in pivoted_train_x.columns if y != '' and z != '' and w != ''] 
    pivoted_train_x.columns = flatten_columns   

    n_components = 37   

    pca = PCA(n_components)
    pca_array = pca.fit_transform(pivoted_train_x)
    pca_df = pd.DataFrame(pca_array, index=pivoted_train_x.index,
                        columns=[f"pca{num+1}" for num in range(n_components)]) 

    result = pd.DataFrame({'설명가능한 분산 비율(고유값)':pca.explained_variance_,
                '기여율':pca.explained_variance_ratio_},
                index=np.array([f"pca{num+1}" for num in range(n_components)]))
    result['누적기여율'] = result['기여율'].cumsum()
    print(result)
    
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms")
    pca_features = pca_df.columns.tolist()
    dend = shc.dendrogram(shc.linkage(pca_df, method='ward', metric='euclidean'), labels=base_features)
    plt.show()
    
    Agg_hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
    y_hc = Agg_hc.fit_predict(pca_df) 

    return y_hc
        
def equips_clustering(df_train_x):
    pivoted_train_x = pd.pivot_table(
        df_train_x,
        values = numeric_features,
        index = 'EQUIP_NIIN',
        columns = ['ITM_NIIN', 'BASE_BIGO', 'ATTRI_YR'],
        )
    pivoted_train_x.reset_index(inplace=True)
    
    pivoted_train_x.dropna(how='all', axis=1, inplace=True) 
    
    col_mean = pivoted_train_x.mean()
    pivoted_train_x.fillna(col_mean, inplace = True)    

    flatten_columns = [str(x) for x, y, z, w in pivoted_train_x.columns if y == '' or z == '' or w == ''] + [str(x) + '_' + str(y) + '_' + str(z) + '_' + str(w) for x, y, z, w in pivoted_train_x.columns if y != '' and z != '' and w != ''] 
    pivoted_train_x.columns = flatten_columns   

    n_components = 15   

    pca = PCA(n_components)
    pca_array = pca.fit_transform(pivoted_train_x)
    pca_df = pd.DataFrame(pca_array, index=pivoted_train_x.index,
                        columns=[f"pca{num+1}" for num in range(n_components)]) 

    result = pd.DataFrame({'설명가능한 분산 비율(고유값)':pca.explained_variance_,
                '기여율':pca.explained_variance_ratio_},
                index=np.array([f"pca{num+1}" for num in range(n_components)]))
    result['누적기여율'] = result['기여율'].cumsum()
    print(result)
    
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms")
    pca_features = pca_df.columns.tolist()
    dend = shc.dendrogram(shc.linkage(pca_df, method='ward', metric='euclidean'), labels=equip_features)
    plt.show()
    
    Agg_hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
    y_hc = Agg_hc.fit_predict(pca_df) 
    print(y_hc)
    
    return y_hc

      
def x_feature_clustering(df_train_x):   
    df_trans_train_x = df_train_x.transpose().iloc[4:,:]

    n_components = 20   

    pca = PCA(n_components)
    pca_array = pca.fit_transform(df_trans_train_x)
    pca_df = pd.DataFrame(pca_array, index=df_trans_train_x.index,
                        columns=[f"pca{num+1}" for num in range(n_components)])

    result = pd.DataFrame({'설명가능한 분산 비율(고유값)':pca.explained_variance_,
                '기여율':pca.explained_variance_ratio_},
                index=np.array([f"pca{num+1}" for num in range(n_components)]))
    result['누적기여율'] = result['기여율'].cumsum()
    print(result)
    
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms")
    dend = shc.dendrogram(shc.linkage(pca_df, method='ward', metric='euclidean'), labels=numeric_features)  
    plt.show()
    
    Agg_hc = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')
    y_hc = Agg_hc.fit_predict(pca_df) 

    return y_hc    
    
if __name__ == "__main__":
    hierachical_clustering(df_train_x)