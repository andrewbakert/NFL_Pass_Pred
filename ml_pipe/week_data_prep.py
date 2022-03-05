import pandas as pd
import numpy as np
import os


def generate_input_data(week1, week2, train=False):
    week_fp = 'nfl-big-data-bowl-2021/week{}.csv'
    play_fp = 'nfl-big-data-bowl-2021/plays.csv'
    week = pd.DataFrame()
    for week_num in range(week1, week2 + 1):
        week_part = pd.read_csv(week_fp.format(week_num))
        week_part['week'] = week_num
        week = pd.concat([week, week_part], axis=0)
    week['time'] = pd.to_datetime(week['time'], format='%Y-%m-%dT%H:%M:%S')
    plays = pd.read_csv(play_fp)
    week_play = week.merge(plays, on=['playId', 'gameId'])
    week_play.sort_values(['gameId', 'playId', 'displayName', 'time'], inplace=True)
    week_play['time_diff'] = week_play.groupby(['playId', 'gameId', 'displayName'])['time'].diff()
    week_play['time_diff'][week_play['time_diff'].isnull()] = pd.Timedelta(0)
    week_play['time_acc_s'] = week_play.groupby(['playId', 'gameId', 'displayName'])['time_diff'].transform(
        lambda x: x.map(lambda x: x.microseconds).cumsum()).div(1e6)
    week_play['yardline_100'] = week_play['absoluteYardlineNumber'].sub(10)

    week_play['x_starting'] = week_play.groupby(['gameId', 'nflId', 'playId'])['x'].transform(lambda x:
                                                                                    x.iloc[0])
    week_play['y_starting'] = week_play.groupby(['gameId', 'playId', 'nflId'])['y'].transform(lambda x:
                                                                                    x.iloc[0])
    week_play['x_behind_line'] = np.where(week_play['x_starting'].gt(week_play['absoluteYardlineNumber']),
                                      week_play['x_starting'].rsub(week_play['absoluteYardlineNumber']),
                                      week_play['x_starting'].sub(week_play['absoluteYardlineNumber']))
    starting_pos_count = (week_play
                          .groupby(['gameId', 'playId', 'nflId'])['position']
                          .first().reset_index()
                          .groupby(['gameId', 'playId', 'position'])['position']
                          .apply(lambda x: x.cumsum())
                          .rename({'position': 'position_num'}, axis=1))
    starting_idx = (week_play
        .groupby(['gameId', 'playId', 'nflId'])
        .first().reset_index()[['gameId', 'playId', 'nflId']])
    starting_idx['position_num'] = starting_pos_count.values
    week_pos = week_play.merge(starting_idx, on=['gameId', 'playId', 'nflId'])
    week_pos['position_num'] = week_pos['position_num'].map(lambda x: x[:2] + str(len(x) // 2))
    off_def = (week_pos
               .groupby(['gameId', 'playId', 'team'])['position']
               .apply(lambda x: 'QB' in x.unique() or 'WR' in x.unique())
               .reset_index()
               .rename({'position': 'off'}, axis=1)
               )
    week_off_def = week_pos.merge(off_def, on=['gameId', 'playId', 'team'])
    week_x_max = week_off_def.groupby(['playId', 'gameId']).apply(lambda x: x[x['off']]['x_starting'].max())
    week_x_max = week_x_max.reset_index().rename({0: 'x_off_max'}, axis=1)
    week_off_def = week_off_def.merge(week_x_max, on=['playId', 'gameId'])
    week_off_def['yardline_first'] = np.where(week_off_def['yardline_100'].gt(week_off_def['x_off_max']),
                                           week_off_def['yardline_100'].add(week_off_def['yardsToGo']),
                                           week_off_def['yardline_100'].sub(week_off_def['yardsToGo']))
    week_off_def['yards_needed_touch'] = np.where(week_off_def['yardline_first'].gt(week_off_def['yardline_100']),
                                                  week_off_def['yardline_first'],
                                                  week_off_def['yardline_first'].rsub(100))
    week_def = week_off_def[week_off_def['off'] == False]
    week_off = week_off_def[week_off_def['off'] == True]
    week_def_starting = (week_def.groupby(['gameId', 'playId', 'position_num'])
    [['x_behind_line', 'y_starting']].first().reset_index()
        .pivot_table(
        columns='position_num',
        values=['x_behind_line', 'y_starting'],
        index=['gameId', 'playId']))
    week_def_starting_cols = ['_'.join(x) for x in week_def_starting.columns]
    week_def_starting.columns = week_def_starting_cols
    week_def_starting.reset_index(inplace=True)
    week_off_starting = week_off.merge(week_def_starting, on=['gameId', 'playId'])
    week_off_starting = week_off_starting.groupby(['nflId', 'playId', 'gameId']).apply(lambda x: x.iloc[1:])
    week_off_starting = week_off_starting[(week_off_starting['x'] != week_off_starting['x_starting']) &
                                          (week_off_starting['y'] != week_off_starting['y_starting'])]
    if not os.path.exists('ml_pipe/chosen_cols.npy') or train is True:
        cols_chosen = ['x', 'y', 'position_num', 'x_behind_line',
                       'y_starting', 'yardsToGo', 'yards_needed_touch', 'time_acc_s', 'yardline_first'] + \
                      week_def_starting_cols
        X_y = week_off_starting[cols_chosen]
        np.save('ml_pipe/chosen_cols.npy', np.array(cols_chosen))
    else:
        cols_chosen = np.load('ml_pipe/chosen_cols.npy', allow_pickle=True)
        X_y = pd.DataFrame()
        for col in cols_chosen:
            try:
                col_val = week_off_starting[col]
            except KeyError:
                col_val = np.nan
            X_y[col] = col_val
    X_y.reset_index(inplace=True, drop=True)
    return week_off_def, week_off_starting, X_y