import pandas as pd
import numpy as np
import pickle

def generate_input_data(week_fp, play_fp, chosen_col_fp=None, for_model=False):
    week = pd.read_csv(week_fp)
    week['time'] = pd.to_datetime(week['time'], format='%Y-%m-%dT%H:%M:%S')
    plays = pd.read_csv(play_fp)
    week_play = week.merge(plays, on=['playId', 'gameId'])
    week_play['time_diff'] = week_play.groupby(['playId', 'gameId', 'displayName'])['time'].diff()
    week_play['time_diff'][week_play['time_diff'].isnull()] = pd.Timedelta(0)
    week_play['time_acc_s'] = week_play.groupby(['playId', 'gameId', 'displayName'])['time_diff'].transform(
        lambda x: x.map(lambda x: x.microseconds).cumsum()).div(1e6)
    week_play['yardline_first'] = np.where(week_play['absoluteYardlineNumber'].gt(week_play['x'].max()),
                                       week_play['absoluteYardlineNumber'].add(week_play['yardsToGo']),
                                       week_play['absoluteYardlineNumber'].sub(week_play['yardsToGo']))
    week_play['x_behind_line'] = np.where(week_play.groupby(['nflId', 'playId'])['x'].transform(
        lambda x: x.iloc[0]).gt(week_play['absoluteYardlineNumber']),
                                      week_play['x'].rsub(week_play['absoluteYardlineNumber']),
                                      week_play['x'].sub(week_play['absoluteYardlineNumber']))
    starting_pos_count = (week_play
                          .groupby(['gameId', 'playId', 'team', 'nflId'])['position']
                          .first().reset_index()
                          .groupby(['gameId', 'playId', 'team', 'position'])['position']
                          .apply(lambda x: x.cumsum())
                          .rename({'position': 'position_num'}, axis=1))
    starting_idx = (week_play
        .groupby(['gameId', 'playId', 'team', 'nflId'])
        .first().reset_index()[['gameId', 'playId', 'team', 'nflId']])
    starting_idx['position_num'] = starting_pos_count.values
    week_pos = week_play.merge(starting_idx, on=['gameId', 'playId', 'team', 'nflId'])
    week_pos['position_num'] = week_pos['position_num'].map(lambda x: x[:2] + str(len(x) // 2))
    week_pos['x_starting_behind_line'] = (week_pos
                                          .groupby(['gameId', 'playId', 'nflId'])['x_behind_line']
                                          .transform(lambda x: x.iloc[0]))
    week_pos['y_starting'] = week_pos.groupby(['gameId', 'playId', 'nflId'])['y'].transform(lambda x: x.iloc[0])
    week_pos['yards_needed_touch'] = np.where(week_pos['absoluteYardlineNumber'].gt(week_pos['yardline_first']),
                                               week_pos['absoluteYardlineNumber'],
                                               week_pos['absoluteYardlineNumber'].rsub(100))
    off_def = (week_pos
               .groupby(['gameId', 'playId', 'team'])['position']
               .apply(lambda x: 'QB' in x.unique() or 'WR' in x.unique())
               .reset_index()
               .rename({'position': 'off'}, axis=1)
               )
    week_off_def = week_pos.merge(off_def, on=['gameId', 'playId', 'team'])
    week_def = week_off_def[week_off_def['off'] == False]
    week_off = week_off_def[week_off_def['off'] == True]
    week_def_starting = (week_def
        .pivot_table(
        columns='position_num',
        values=['x_starting_behind_line', 'y_starting'],
        index=['gameId', 'playId']))
    week_def_starting_cols = ['_'.join(x) for x in week_def_starting.columns]
    week_def_starting.columns = week_def_starting_cols
    week_def_starting.reset_index(inplace=True)
    week_off_starting = week_off.merge(week_def_starting, on=['gameId', 'playId'])
    week_off_starting = week_off_starting.groupby(['nflId', 'playId', 'gameId']).apply(lambda x: x.iloc[1:])
    if chosen_col_fp == None:
        cols_chosen = ['x', 'y', 'position_num', 'x_starting_behind_line',
                       'y_starting', 'yardsToGo', 'yards_needed_touch', 'time_acc_s', 'yardline_first'] + \
                      week_def_starting_cols
        X_y = week_off_starting[cols_chosen]
        if for_model == True:
            np.save('ml_pipe/chosen_cols.npy', np.array(cols_chosen))
    else:
        cols_chosen = np.load(chosen_col_fp, allow_pickle=True)
        X_y = pd.DataFrame()
        for col in cols_chosen:
            try:
                col_val = week_off_starting[col]
            except KeyError:
                col_val = np.nan
            X_y[col] = col_val
    X_y.reset_index(inplace=True, drop=True)
    return week_off_starting, X_y