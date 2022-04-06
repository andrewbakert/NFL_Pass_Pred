# IMPORTING LOCAL MODULES
import importlib
import form_pred
import ball_movement
import get_data
import def_clean
import TrainTestNFL

# REFRESHING LOCAL CHANGES
importlib.reload(get_data)
importlib.reload(form_pred)
importlib.reload(def_clean)
importlib.reload(ball_movement)
importlib.reload(TrainTestNFL)

# IMPORTING LOCAL PACKAGES
from get_data import get_assets, get_positional_data
from form_pred import clean_positional
from ball_movement import ball_quadrants, make_quad_chart
from def_clean import DefensiveCleaning
from TrainTestNFL import TrainTestNFL
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans

class PrepPipe:
    def __init__(self, first=1, last=14, n_cuts=11, frameLimit=11,
                 simMethod='distance', quad_num=4, def_fp='assets/def_clean_output.csv'):
        self.first = first
        self.last = last
        self.n_cuts = n_cuts
        self.frameLimit = frameLimit
        self.simMethod = simMethod
        self.quad_num = quad_num
        self.def_fp = def_fp
        if not os.path.exists('Kaggle-Data-Files'):
            get_assets()
        self.positions = get_positional_data()

    def clean_data(self):
        quads = ball_quadrants(self.positions,self.quad_num)
        offense = clean_positional(self.positions)
        try:
            defense = pd.read_csv(self.def_fp).reset_index()
            if 2018123015 not in defense['gameId'].to_list():
                print('missing full 17 week dataset')
                print('getting dataset now.')
                raise LookupError
        except (FileNotFoundError, LookupError):
            def_cleaning = DefensiveCleaning(weeks_data=self.positions, n_cuts=self.n_cuts,
                                             frameLimit=self.frameLimit, simMethod=self.simMethod,
                                             )
            defense = def_cleaning.generate_full_df(1, 17, fp=self.def_fp).reset_index()
        self.train_test = TrainTestNFL(offense,defense,quads)
        X_train, X_test, y_train, y_test = self.train_test.split(self.first, self.last)
        y_train = y_train[['x_quad', 'y_quad']]
        y_test = y_test[['x_quad', 'y_quad']]
        return X_train, X_test, y_train, y_test


class OffensiveFormation(BaseEstimator, TransformerMixin):
    def __init__(self, model=False, model_params=False, model_fp='models/off_form.pkl',
                 cv=5, scoring='f1_micro'):
        if not model:
            self.model = LogisticRegression(max_iter=10000)
        else:
            self.model = model
        if not model_params:
            self.model_params = {'C': [10**x for x in range(-4, 4)]}
        else:
            self.model_params = model_params
        self.model_fp = model_fp
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y=None):
        if self.model_fp and os.path.exists(self.model_fp):
            with open(self.model_fp, 'rb') as model:
                self.grid = pickle.load(model)
            X_train = X.drop('offenseFormation', axis=1)
            self.X_cols = X_train.columns
            self.scaler = StandardScaler()
            self.scaler.fit_transform(X_train)
        else:
            self.grid = GridSearchCV(self.model, param_grid=self.model_params, cv=self.cv,
                                     scoring=self.scoring)
            X_train = X.drop('offenseFormation', axis=1)
            X_train_in = pd.DataFrame()
            for col in self.X_cols:
                if col in X_train.columns:
                    X_train_in[col] = X_train[col]
                else:
                    X_train[col] = 0
            y_train = X['offenseFormation']
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_in)
            self.grid.fit(X_train_scaled, y_train)
            base = self.model_fp.split('/')[0]
            if self.model_fp:
                if not os.path.exists(base):
                    os.mkdir(base)
                with open(self.model_fp, 'wb') as model:
                    pickle.dump(self.grid, model)
        print("Offensive formation model fitted")
        return self

    def transform(self, X):
        X = X.drop('offenseFormation', axis=1)
        X_scaled = self.scaler.transform(X)
        y = self.grid.predict(X_scaled)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_df['offenseFormation'] = y
        print("Offensive formation predicted")
        return X_scaled_df


class DefensiveClustering(BaseEstimator, TransformerMixin):
    def __init__(self, cols='all', n_clusters=5, pca_variance=0.8):
        self.cols = cols
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance

    def fit(self, X, y=None):
        actions = [action for action in X.columns if '_act' in action]
        self.melt_cols = ['gameId','playId'] + actions

        melt_df = X[self.melt_cols]
        melt_df = melt_df.melt(['gameId','playId']).dropna()
        melt_df = melt_df.groupby(['gameId','playId','value']).count()
        melt_df = melt_df.reset_index().pivot(index=['gameId','playId'],
                                              columns='value',values='variable').fillna(0)
        melt_df['TOT'] = melt_df['B'] + melt_df['M'] + melt_df['Z']
        melt_df['%B'] = melt_df['B'] / melt_df['TOT']
        melt_df['%M'] = melt_df['M'] / melt_df['TOT']
        melt_df['%Z'] = melt_df['Z'] / melt_df['TOT']
        melt_df = melt_df.fillna(0)

        self.orig_cols =  ['gameId','playId','defendersInTheBox','numberOfPassRushers','DB','LB','DL',
                           'yardline_first_dir','yardline_100_dir']
        orig_df = X[self.orig_cols].set_index(['gameId','playId'])

        orig_df = orig_df.merge(melt_df[['%B','%M','%Z']], on=['gameId','playId']).fillna(0)
        print(orig_df)
        if self.cols != 'all':
            orig_df = orig_df[self.cols]
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(orig_df)
        self.pca = PCA(n_components=self.pca_variance)
        scores_pca = self.pca.fit_transform(X)
        self.kmeans_pca = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        self.kmeans_pca.fit(scores_pca)
        print('KMeans and PCA fitted')
        return self


    def transform(self, X):
        melt_df = pd.DataFrame()
        for col in self.melt_cols:
            if col in X.columns:
                melt_df[col] = X[col]
            else:
                melt_df[col] = np.nan
        melt_df = melt_df.melt(['gameId','playId']).dropna()
        melt_df = melt_df.groupby(['gameId','playId','value']).count()
        melt_df = melt_df.reset_index().pivot(index=['gameId','playId'],
                                              columns='value',values='variable').fillna(0)
        melt_df['TOT'] = melt_df['B'] + melt_df['M'] + melt_df['Z']
        melt_df['%B'] = melt_df['B'] / melt_df['TOT']
        melt_df['%M'] = melt_df['M'] / melt_df['TOT']
        melt_df['%Z'] = melt_df['Z'] / melt_df['TOT']
        melt_df = melt_df.fillna(0)
        orig_df = X[self.orig_cols].set_index(['gameId','playId'])
        orig_df = orig_df.merge(melt_df[['%B','%M','%Z']], on=['gameId','playId']).fillna(0)
        if self.cols != 'all':
            orig_df = orig_df[self.cols]
        X = self.scaler.transform(orig_df)

        scores_pca = self.pca.transform(X)
        kmeans_vals = self.kmeans_pca.transform(scores_pca)
        kmeans_vals_df = pd.DataFrame(kmeans_vals, columns=[f'cluster_{i}'
                                                            for i in range(kmeans_vals.shape[1])])
        pca_df = pd.DataFrame(scores_pca, columns=[f'pc_{i}' for i in range(scores_pca.shape[1])])
        df_seg = pd.concat([orig_df.reset_index()[['gameId','playId']], kmeans_vals_df, pca_df], axis=1)
        df_seg['cluster'] = self.kmeans_pca.labels_
        df_seg.drop(['gameId', 'playId'], axis=1, inplace=True)
        print("Defensive position transformed")

        return df_seg