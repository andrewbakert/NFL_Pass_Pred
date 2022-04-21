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
from ball_movement import ball_quadrants
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import re

class PrepPipe:
    """
    Processes data using defensive and offensive pipelines and splits into train and test sets

    Attributes
    ----------
    first : int
        Starting week for train set
        default 1

    last : int
        Ending week for train set
        default : 14

    n_cuts : int
        Number of cuts in play used to average defensive positions
        default : 11

    frameLimit : int
        Minimum number of 100 ms periods in play. Should be the same as n_cuts.
        default : 11

    simMethod : str
        Method of evaluating similarity between players. Used to evaluate defensive position.
        default : 'distance'

    quad_num : int
        Number of x and y quadrants to break decision space into.
        default : 4

    def_fp : str
        Filepath for processed defensive data
        default : 'assets/def_clean_output.csv'

    Methods
    -------
    clean_data
        Cleans all data and breaks into train and test set
    """
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
        """
        Cleans data and returns train and test set
        :return: tuple of train and test sets, both X and y.
        """

        # Create quadrants
        quads = ball_quadrants(self.positions,self.quad_num)

        # Clean defensive data
        offense = clean_positional(self.positions)

        # Check if there is already a defensive file and that it contains all games.
        # If so, load from file. If not, clean defensive data.
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

        # Instantiate class to break up into sets.
        self.train_test = TrainTestNFL(offense,defense,quads)
        X_train, X_test, y_train, y_test = self.train_test.split(self.first, self.last)

        # Choose only quadrants for y.
        y_train = y_train[['x_quad', 'y_quad']]
        y_test = y_test[['x_quad', 'y_quad']]
        return X_train, X_test, y_train, y_test


class OffensiveFormation(BaseEstimator, TransformerMixin):
    """
    Creates SKLearn transformer that predicts the offensive formation based on starting positions.

    Attributes
    ----------
    model : bool or SKLearn model
        Flag for whether model is specified, or specified model.
        default : False

    model_params : bool or dict
        Flag for whether grid search parameters are specified or parameters
        default : False

    model_fp : str
        Filepath for existing model if available.
        default : 'models/off_form.pkl'

    scaler_fp : str
        Filepath for existing scaler if available
        default : "models/off_scaler.pkl"

    cv : int
        Number of cross validation folds used in grid search
        default : 5

    scoring : str
        Scoring parameter used to evaluate
        default : "fi_micro"

    Methods
    -------
    fit
        Fit grid search and scaler

    transform
        Add in offensive formation prediction

    """
    def __init__(self, model=False, model_params=False, model_fp='models/off_form.pkl',
                 scaler_fp="models/off_scaler.pkl",
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
        self.scaler_fp = scaler_fp
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y=None):
        """
        Fit grid search and scaler

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe used to fit models

        y : pd.Series
            Y data
            default : None

        Returns
        -------
        self
        """

        # Check if model and scaler already saved. If so, load with pickle.
        if self.model_fp and os.path.exists(self.model_fp):
            with open(self.model_fp, 'rb') as model:
                self.grid = pickle.load(model)
            with open(self.scaler_fp, "rb") as scaler:
                self.scaler = pickle.load(scaler)
            X_train = X.drop('offenseFormation', axis=1)
            self.X_cols = X_train.columns
        else:
            # Instantiate grid with parameters and model
            self.grid = GridSearchCV(self.model, param_grid=self.model_params, cv=self.cv,
                                     scoring=self.scoring)

            # Select all but offensive formation for prediction.
            X_train = X.drop('offenseFormation', axis=1)
            y_train = X['offenseFormation']

            # Instantiate scaler and transform
            self.scaler = StandardScaler()
            self.X_cols = X_train.columns
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Fit grid with scaled data
            self.grid.fit(X_train_scaled, y_train)
            base = self.model_fp.split('/')[0]

            # Save model if newly trained
            if self.model_fp:
                if not os.path.exists(base):
                    os.mkdir(base)
                with open(self.model_fp, 'wb') as model:
                    pickle.dump(self.grid, model)
        return self

    def transform(self, X):
        X = X.drop('offenseFormation', axis=1)
        X_in = pd.DataFrame()
        for col in self.X_cols:
            if col in X.columns:
                X_in[col] = X[col]
            else:
                X_in[col] = 0
        X_scaled = self.scaler.transform(X_in)
        y = self.grid.predict(X_scaled)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_df['offenseFormation'] = y
        return X_scaled_df


class DefensiveClustering(BaseEstimator, TransformerMixin):
    def __init__(self, cols='all', n_clusters=5, pca_variance=0.8):
        self.cols = cols
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance

    def fit(self, X, y=None):
        actions = [action for action in X.columns if '_act' in action]
        self.melt_cols = ['gameId','playId'] + actions

        melt_df = X[self.melt_cols].copy()
        melt_df = melt_df.melt(['gameId','playId'])
        melt_df = melt_df[melt_df['value'] != 0]
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
        orig_df = X[self.orig_cols].copy()

        orig_df = orig_df.merge(melt_df[['%B','%M','%Z']], on=['gameId','playId']).fillna(0)
        if self.cols != 'all':
            orig_df = orig_df[self.cols]
        else:
            orig_df.drop(['gameId', 'playId'], axis=1, inplace=True)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(orig_df)
        self.pca = PCA(n_components=self.pca_variance)
        scores_pca = self.pca.fit_transform(X_scaled)
        self.kmeans_pca = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        kmeans_vals = self.kmeans_pca.fit_transform(scores_pca)
        self.kmeans_scaler = StandardScaler()
        self.kmeans_scaler.fit(kmeans_vals)
        return self


    def transform(self, X, y=None):
        melt_df = pd.DataFrame()
        for col in self.melt_cols:
            if col in X.columns:
                melt_df[col] = X[col].copy()
            else:
                melt_df[col] = np.nan
        melt_df = melt_df.melt(['gameId','playId']).dropna()
        melt_df = melt_df[melt_df['value'] != 0]
        melt_df = melt_df.groupby(['gameId','playId','value']).count()
        melt_df = melt_df.reset_index().pivot(index=['gameId','playId'],
                                              columns='value',values='variable').fillna(0)
        melt_df['TOT'] = melt_df['B'] + melt_df['M'] + melt_df['Z']
        melt_df['%B'] = melt_df['B'] / melt_df['TOT']
        melt_df['%M'] = melt_df['M'] / melt_df['TOT']
        melt_df['%Z'] = melt_df['Z'] / melt_df['TOT']
        melt_df = melt_df.fillna(0)
        orig_df = X[self.orig_cols].copy()
        orig_df = orig_df.merge(melt_df[['%B','%M','%Z']], on=['gameId','playId']).fillna(0)
        if self.cols != 'all':
            orig_df = orig_df[self.cols]
        else:
            orig_df.drop(['gameId', 'playId'], axis=1, inplace=True)
        X_scaled = self.scaler.transform(orig_df)

        scores_pca = self.pca.transform(X_scaled)
        kmeans_vals = self.kmeans_pca.transform(scores_pca)
        kmeans_vals_scaled = self.kmeans_scaler.transform(kmeans_vals)
        kmeans_vals_df = pd.DataFrame(kmeans_vals_scaled, columns=[f'cluster_{i}'
                                                                   for i in range(kmeans_vals.shape[1])])
        pca_df = pd.DataFrame(scores_pca, columns=[f'pc_{i}' for i in range(scores_pca.shape[1])])
        X = pd.concat([orig_df, kmeans_vals_df, pca_df], axis=1)
        X['cluster'] = self.kmeans_pca.predict(scores_pca)
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]

class SideNotValidError(Exception):
    def __init__(self, message="Side not valid. Choose 'off', 'def', or 'both'"):
        self.message = message
        super().__init__(self.message)


class FullPipeWrapper(PrepPipe):
    def __init__(self, first=1, last=14, n_cuts=11, frameLimit=11,
                 simMethod='distance', quad_num=4, def_fp='assets/def_clean_output.csv'):
        super().__init__(first, last, n_cuts, frameLimit, simMethod,
                         quad_num, def_fp)

    def extract_data_cols(self):
        self.X_train, self.X_test, y_train, y_test = self.clean_data()
        self.y_train_x = y_train.iloc[:, 0]
        self.y_train_y = y_train.iloc[:, 1]
        self.y_test_x = y_test.iloc[:, 0]
        self.y_test_y = y_test.iloc[:, 1]
        self.off_col = self.train_test.ofc.drop(['gameId', 'playId', 'gamePlayId', 'week'], axis=1).columns
        self.off_col = self.off_col.append(pd.Index(['yardline_first_dir', 'yardline_100_dir']))
        self.def_col = self.train_test.dfc.drop(['week'], axis=1).columns
        form_idx = self.off_col.get_loc('offenseFormation')
        self.off_info_cols = self.off_col[form_idx+1:]
        self.off_form_cols = self.off_col[:form_idx+1]
        self.off_cat_info_cols = ['down', 'possessionTeam']
        self.off_info_cols = self.off_info_cols[~np.isin(self.off_info_cols, self.off_cat_info_cols)]
        self.def_start_col = [col for col in self.def_col if re.search('start', col)]
        self.def_start_col_y = [col for col in self.def_start_col if re.search('y', col)]
        self.def_start_col_x = [col for col in self.def_start_col if re.search('x', col)]

    def build_pipe(self, side='both', model=LogisticRegression()):
        if not hasattr(self, "X_train"):
            self.extract_data_cols()
        off_pre_one_pipe = ColumnTransformer([
            ('info_scale', StandardScaler(), self.off_info_cols),
            ('form', OffensiveFormation(), self.off_form_cols),
            ('filter_cols', FunctionTransformer(lambda x: x), self.off_cat_info_cols)
        ])
        off_pre_one_add_col = Pipeline([('off_pre_one', off_pre_one_pipe),
                                        ('func_trans', FunctionTransformer(lambda x:
                                                                           pd.DataFrame(x,
                columns=list(self.off_info_cols) + list(self.off_form_cols) + list(self.off_cat_info_cols)))),
                                        ('select_cols', FeatureSelector(
                    list(self.off_info_cols) + list(self.off_form_cols) + list(self.off_cat_info_cols)))])

        form_one_pipe = ColumnTransformer([('off_form_one', OneHotEncoder(handle_unknown='ignore'),
                                self.off_cat_info_cols + ['offenseFormation'])], remainder='passthrough')
        off_full_pipe = Pipeline([('full_cols', off_pre_one_add_col), ('one_hot', form_one_pipe)])

        def_one_pipe = ColumnTransformer([('def_clust_one', OneHotEncoder(handle_unknown='ignore'),
                                           [-1])], remainder='passthrough')
        def_pass = Pipeline([('select_cols', FeatureSelector(self.def_start_col)),
                             ('pass', FunctionTransformer(lambda x: x))])
        def_clust = ColumnTransformer([
            ('pass', def_pass, self.def_start_col),
            ('def_clust', DefensiveClustering(), self.def_col)
                                    ])
        def_full_pipe = Pipeline([
            ('def_clust_pass', def_clust),
            ('def_clust_one', def_one_pipe)])

        full_pipe = ColumnTransformer([('off', off_full_pipe, self.off_col),
                                       ('def', def_full_pipe, self.def_col)])

        if side == 'off':
            # Offensive pipeline alone
            pipe = Pipeline([('select_cols', FunctionTransformer(lambda x: x[self.off_col])),
                             ('off_full_pipe', off_full_pipe), ('model', model)])
        elif side == 'def':
            pipe = Pipeline([('select_cols', FunctionTransformer(lambda x: x[self.def_col])),
                             ('def_full_pipe', def_full_pipe),
                             ('model', model)])
        elif side == 'both':
            pipe = Pipeline([('full_pipe', full_pipe),
                             ('to_float', FunctionTransformer(lambda x: x.astype(float))),
                             ('model', model)])
        else:
            raise SideNotValidError
        return pipe