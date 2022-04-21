import os
import pickle

class TrainTestNFL:
    """Class to create a train test split with NFL data"""
    def __init__(self,off = None, dfc = None, y = None):
        self.ofc = off
        self.dfc = dfc
        self.y = y
        self.checks = []
            
    def split(self, first = 1, last = 17):
        """Split given Week {first} and Week {Last} for Train / Test Split

        Parameters
        ----------
        first : int
            First week (inclusive) to split

        last : int
            Last week (inclusive) of split """
        #imports
        import pandas as pd
        import numpy as np
        games = pd.read_csv('nfl-big-data-bowl-2021/games.csv')
        
        # Checks for usable data
        if "week" in self.ofc.columns:
            pass
        elif "gameId" in self.ofc.columns:
            self.ofc = self.ofc.merge(games[['gameId','week']], left_on = 'gameId', right_on = 'gameId')
        else:
            self.checks.append('offense')

        if "week" in self.dfc.columns:
            pass
        elif "gameId" in self.dfc.columns:
            self.dfc = self.dfc.merge(games[['gameId','week']], left_on = 'gameId', right_on = 'gameId')
        else:
            self.checks.append('defense')

        if "week" in self.y.columns:
            pass
        elif "gameId" in self.y.columns:
            self.y = self.y.merge(games[['gameId','week']], left_on = 'gameId', right_on = 'gameId')
        else:
            self.checks.append('ball placement (y)')
        if len(self.checks) > 0:
            raise Exception(f'The following dataframes don\'t have a week or gameId for split: {self.checks}')

        #Merge Offense and Defense to form X
        self.X = pd.merge(self.ofc, self.dfc, how='inner', left_on = ['gameId','playId','week'],
                          right_on = ['gameId','playId','week']).fillna(0)

        #merge y with X gameIds so the data matches up
        self.y = pd.merge(self.y, self.X[['gameId','playId','week']], how='inner', left_on = ['gameId','playId','week'], right_on = ['gameId','playId','week']).fillna(0)

        #Merge X back with Y so the data matches up
        self.X = pd.merge(self.X, self.y[['gameId','playId','week']], how='inner', left_on = ['gameId','playId','week'], right_on = ['gameId','playId','week']).fillna(0)

    # Checks for start and last
        if first not in self.X['week'].to_list():
            raise Exception(f'Starting week {first} not in df')  

        if last not in self.X['week'].to_list(): 
            raise Exception(f'Ending week {last} not in df')  
    #If not train/test splitting
        if first == 1 and last == 17:
            self.X_train = self.X
            self.X_test = None
            self.y_train = self.y
            self.y_test = None
    #if splitting
        else:
            self.X_train = self.X[(self.X['week'] >= first) & (self.X['week'] <= last)]
            self.X_test = self.X[(self.X['week'] > last)]
            self.y_train = self.y[(self.y['week'] >= first) & (self.y['week'] <= last)]
            self.y_test = self.y[(self.y['week'] > last)]
        return self.X_train, self.X_test, self.y_train, self.y_test

                

    def formation_pred(self, confusion_matrix = False):
        """Replace the True Offensive Formation labels with the predicted offensive formation labels

        Parameters
        ----------
        confusion_matrix : bool
            print a confusion matrix. Default == False """

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import GridSearchCV

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train[self.ofc.columns].drop('offensiveFormation', axis = 1))
        X_test_scaled = scaler.transform(self.X_test[self.ofc.columns].drop('offensiveFormation', axis = 1))

        log_reg = LogisticRegression(max_iter=10000)
        cross_val_score(log_reg, X_train_scaled, self.X_train['offensiveFormation'], cv=5)

        params_lr = {'C': [10**x for x in range(-4, 4)]}
        grid_lr = GridSearchCV(log_reg, params_lr, cv=3, scoring='f1_micro')
        grid_lr.fit(X_train_scaled, self.X_train['offensiveFormation'])
        print('Model best score:', grid_lr.best_score_)

        self.X_test['offenseFormationPrediction'] = grid_lr.predict(X_test_scaled)
        self.X_train['offenseFormationPrediction'] = grid_lr.predict(X_train_scaled)

        if confusion_matrix:
            y_pred = grid_lr.predict(self.X_test_scaled)
            from sklearn.metrics import classification_report
            print(classification_report(self.X_test['offensiveFormation'], y_pred))
            print(confusion_matrix(self.self.X_test['offensiveFormation'], y_pred))
        
        return self.X_train, self.X_test, self.y_train, self.y_test