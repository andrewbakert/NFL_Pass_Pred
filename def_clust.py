# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


#df = pd.read_csv('assets/def_clean_output.csv')

# %%
def optimize_pca_components(df):
    pca = PCA()
    pca.fit(df)
    evr = pca.explained_variance_ratio_.cumsum()
    n_comps = len(np.argwhere(evr <= 0.99)) + 1
    n_comps_opt = len(np.argwhere(evr <= 0.8)) + 1
    plt.figure(figsize=(6,6))
    plt.plot(range(1,n_comps+1), evr[:n_comps], marker='o', linestyle='--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    print('The optimal number of components is approximately ', n_comps_opt)
    return n_comps_opt

# %%
def optimize_kmeans_clusters_with_pca(df, n_comps):
    pca = PCA(n_components=n_comps)
    pca.fit(df)
    comps_pca = pca.components_
    scores_pca = pca.transform(df)
    wcss = []
    sil = []
    
    for i in range(2,21):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(scores_pca)
        y_pred = kmeans_pca.predict(scores_pca)
        wcss.append(kmeans_pca.inertia_)

        sscore = metrics.silhouette_score(scores_pca, y_pred)

        sil.append(sscore)
        
    plt.figure(figsize=(6,6))
    plt.plot(range(2,21),wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with PCA Clustering - Elbow Test')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.plot(range(2,21),sil, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('K-means with PCA Clustering - Silhouette Score Curve')
    plt.show()
    
    return scores_pca, comps_pca

# %%
def kmeans_clusters_and_dataframe(scores_pca, df, n_clusters, filename):
    kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    df_seg = pd.concat([df.reset_index()[['gameId','playId']],pd.DataFrame(scores_pca)], axis=1)
    n = int(scores_pca.shape[1])
    
    comp_labels = []

    for i in range(1,n+1):
        comp_labels.append('Component '+ str(i))

    seg_list = ['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth']
    seg_dict = {}
    
    for i in range(0,n_clusters):
         seg_dict[i] = seg_list[i]
    
    df_seg.columns.values[-n:] = comp_labels
    df_seg['Cluster Type'] = kmeans_pca.labels_
    df_seg['Cluster'] = df_seg['Cluster Type'].map(seg_dict)
    
    print("The component names are as follows: ", comp_labels)
    path = 'assets/' + filename
    
    df_seg.to_csv(path)
    
    return df_seg

# %%
def prep_data(df, scale=True):
    actions = [action for action in df.columns if '_act' in action]
    melt_cols = ['gameId','playId'] + actions

    melt_df = df[melt_cols]
    melt_df = melt_df.melt(['gameId','playId']).dropna()
    melt_df = melt_df.groupby(['gameId','playId','value']).count()
    melt_df = melt_df.reset_index().pivot(index=['gameId','playId'],columns='value',values='variable').fillna(0)
    melt_df['TOT'] = melt_df['B'] + melt_df['M'] + melt_df['Z']
    melt_df['%B'] = melt_df['B'] / melt_df['TOT']
    melt_df['%M'] = melt_df['M'] / melt_df['TOT']
    melt_df['%Z'] = melt_df['Z'] / melt_df['TOT']
    melt_df = melt_df.fillna(0)

    orig_cols =  ['gameId','playId','defendersInTheBox','numberOfPassRushers','DB','LB','DL','yardline_first','yardline_100']
    orig_df = df[orig_cols].set_index(['gameId','playId'])

    orig_df = orig_df.merge(melt_df[['%B','%M','%Z']], on=['gameId','playId']).fillna(0)
    
    if scale == True:
        X = StandardScaler().fit_transform(orig_df)
    else:
        X = np.array(orig_df)
    

    return X, orig_df

# %%
def plot_pca_heatmap(pca, df):
    pca_names = []
    cols = list(df.columns)
    num = int(pca.shape[0])+1
    
    for i in range(1,num):
        pca_names.append('PC #' + str(i))

    #max_f = len(cols) - 1
    f_ticks = len(cols) - 1
    c_ticks = num -1
    
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.imshow(pca, interpolation = 'none', cmap = 'plasma')
    plt.xticks(np.arange(-0.1,  f_ticks, 1) , cols, rotation = 75, fontsize=12)
    plt.yticks(np.arange(0.0, c_ticks, 1), pca_names, fontsize = 16)
    plt.colorbar()
    plt.savefig('assets/pca_plot.png')
    

# %%
def kmeans_visual(df, comp_x, comp_y):

    x_axis = df[comp_x]
    y_axis = df[comp_y]
    plt.figure(figsize=(10,8))
    sns.color_palette("husl", 9)
    sns.scatterplot(x_axis, y_axis, hue = df['Cluster'])
    plt.title('Clusters by PCA Components')
    plt.show()

# %%
def return_pca_and_clusters(df, n_clusters=5):

    X, pca_df = prep_data(df, scale=True)
    n_comps = optimize_pca_components(X)
    scores_pca, comps_pca = optimize_kmeans_clusters_with_pca(X, n_comps)
    plot_pca_heatmap(comps_pca, pca_df)
    df_seg = kmeans_clusters_and_dataframe(scores_pca, pca_df, n_clusters, 'def_clust_output.csv')
    
    return df_seg
