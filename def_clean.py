# %%


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from get_data import get_assets, get_positional_data
import time
import warnings
warnings.filterwarnings('ignore')

# %%

class InvalidSimilarityMetricError(Exception):
    """
    Exception for errors in similarity metric chosen

    Attributes
    ----------
    message : str
        explanation of the error
    """
    def __init__(self, salary, message='Invalid similarity metric. Must be "distance" or "cosine"'):
        self.message = message
        super().__init__(self.message)

class DefensiveCleaning:
    def __init__(self, n_cuts, frameLimit=11, simMethod='distance'):
        print('..............................initializing')
        if not os.path.exists('Kaggle-Data-Files'):
            get_assets()
        self.weeks_data = get_positional_data()
        self.play_data = pd.read_csv('Kaggle-Data-Files/plays.csv')
        self.game_data = pd.read_csv('Kaggle-Data-Files/games.csv')
        print('..data downloaded...')
        self.n_cuts = n_cuts
        self.frameLimit = frameLimit
        self.simMethod = simMethod

    def new_time(self, week):
        #df = df[df['playId']==2372]
        df = self.weeks_data[self.weeks_data['week'] == week]
        print('...Week {} loaded...'.format(str(week)))
        time = df[['gameId','playId','time']]

        df['key'] = df['gameId'].astype(str) + df['playId'].astype(str) + df['time'].astype(str) + df['nflId'].astype(str)

        man_ms = df.groupby('key').cumcount()

        fb_df = df[df['nflId'].isnull()]

        tot_ms = fb_df[['gameId','playId','time','event']].groupby(by=['gameId','playId','time']).count()

        time_df = pd.concat([time, man_ms], axis=1).rename(columns={0:'new_sec'})

        tot_ms = tot_ms.reset_index()

        time_df = time_df.reset_index().merge(tot_ms, on=['gameId','playId','time']).set_index('index')

        time_df['new_sec'] = 10 - time_df['event'] + time_df['new_sec']

        time_df['check'] = time_df.time.str[20:21]

        time_df['man_time'] = time_df.time.str[:20] + time_df['new_sec'].astype(str) + '00Z'

        time_df['new_time'] = np.where(time_df['check']!=time_df['new_sec'], time_df['man_time'], time_df['time'])

        time = time_df['new_time']

        df = df.reset_index().merge(time, on='index').set_index('index').drop(columns=['key','time']).rename(columns={'new_time':'time'})
        print('...accumulated time caluclated...')
        return df

    def filter_full_position_df(self, week):
        #df = self.new_time(week=week)
        df = self.weeks_data[self.weeks_data['week'] == week]
        print('...Week {} loaded...'.format(str(week)))
        fb_df = df[df['nflId'].isnull()]
        pos_df = df[df['nflId'].notnull()]

        fb_df['time'] =  pd.to_datetime(fb_df['time'], format='%Y-%m-%dT%H:%M:%S')

        # Find time that pass was thrown and merge with main df.
        pass_start = fb_df[fb_df['event'] == 'pass_forward'][['gameId', 'playId', 'frameId']].rename({'frameId': 'frame_pass'}, axis=1)

        ball_snap = fb_df[fb_df['event'] == 'ball_snap'][['gameId', 'playId', 'frameId']].rename({'frameId': 'frame_snap'}, axis=1)

        pos_df = pos_df.merge(pass_start, on=['gameId', 'playId'], how='left')
        pos_df = pos_df.merge(ball_snap, on=['gameId', 'playId'], how='left')

        # Convert time to datetime format.
        pos_df['time'] = pd.to_datetime(pos_df['time'], format='%Y-%m-%dT%H:%M:%S')

        # Find whether part of play was before pass.
        pos_df['before_pass'] = pos_df['frameId'].le(pos_df['frame_pass'])
        pos_df['after_snap'] = pos_df['frameId'].ge(pos_df['frame_snap'])

        # Filter to include only part of play before pass.
        pos_df = pos_df[pos_df['before_pass']]
        pos_df = pos_df[pos_df['after_snap']]

        uniq_df = pos_df[['frameId','gameId','playId','event']]

        uniq_df = uniq_df.groupby(by=['frameId','gameId','playId']).count().reset_index().drop(columns='event')

        uniq_df = uniq_df.groupby(by=['gameId','playId']).count().sort_values(by='frameId').rename(columns={'frameId':'frameCount'})

        pos_df = pos_df.merge(uniq_df.reset_index(), on=['gameId','playId'])

        short_df = pos_df[pos_df['frameCount'] < self.frameLimit]
        short_df = short_df[['gameId','playId','frameId']].groupby(by=['gameId','playId']).count().drop(columns='frameId')

        #short_df.to_csv('assets/short_plays.csv')

        pos_df = pos_df[pos_df['frameCount'] >= self.frameLimit]

        print('...filtered...')
        return pos_df.drop(columns=['before_pass','after_snap'])

    def plot_histogram(self, df, column, bins, hue, fig_name):
        sns.histplot(data=df, x=column, bins=bins, hue=hue)
        figure = 'assets/' + fig_name + '.png'
        plt.savefig(figure)
        plt.close()

    def transform_directions(self, week):
        df = self.filter_full_position_df(week=week)
        play_data = self.play_data.copy()
        df['time_diff'] = df.groupby(['playId', 'gameId', 'displayName'])['time'].diff()
        df['time_diff'][df['time_diff'].isnull()] = pd.Timedelta(0)
        df['time_acc_s'] = df['time_diff'].dt.microseconds.div(1e6)
        df['time_acc_s'] = df.groupby(['playId', 'gameId', 'nflId'])['time_acc_s'].transform("cumsum")
        df.drop('time_diff', axis=1, inplace=True)

        play_df = play_data[play_data['absoluteYardlineNumber'].notnull()]
        #print(play_df.shape)
        play_df = play_df[['gameId','playId', 'absoluteYardlineNumber','yardsToGo','personnelD',
                           'defendersInTheBox','numberOfPassRushers', 'possessionTeam']]

        # Merge movement and play-by-play datasets.
        df = df.merge(play_df, on=['gameId', 'playId']).merge(
            self.game_data[['gameId', 'visitorTeamAbbr', 'homeTeamAbbr']], on='gameId')

        # Find which teams in the dataframe are offensive vs. defensive.
        df['off'] = np.where(((df['team'] == 'away') &
                                 (df['possessionTeam'] == df['visitorTeamAbbr'])) | ((df['team'] == 'home') &
                                 (df['possessionTeam'] == df['homeTeamAbbr'])),
                                 True, False)

        df.drop(['possessionTeam', 'visitorTeamAbbr', 'homeTeamAbbr'], axis=1, inplace=True)

        #print(df.columns)
        #print("check for any offensive positions not mapped", df[~df['off']]['position'].unique())

        # Extract starting x and y position.
        df['x_starting'] = df.groupby(['gameId', 'playId', 'nflId'])['x'].transform("first")
        df['y_starting'] = df.groupby(['gameId', 'playId', 'nflId'])['y'].transform("first")

        # Subtract 10 from yardline to get relative to left endzone.
        df['yardline_100'] = df['absoluteYardlineNumber'].sub(10)

        # Extract data for offense, including yardline numbers. Used to find the yardline for the first down.
        off_df = df[df['off']].groupby(['gameId', 'playId'])[[
            'x_starting', 'yardline_100', 'absoluteYardlineNumber', 'yardsToGo']].first().reset_index()
        off_df['yardline_first'] = np.where(off_df['x_starting'].gt(off_df['absoluteYardlineNumber']),
                                            off_df['yardline_100'].sub(off_df['yardsToGo']),
                                            off_df['yardline_100'].add(off_df['yardsToGo']))

        # Merge main dataframe with dataframe containing the first down yardline.
        # Then extract which side offense is on.
        df = df.merge(off_df[['gameId', 'playId', 'yardline_first']].drop_duplicates(), on=['gameId', 'playId'])
        df['off_dir'] = np.where(df['yardline_first'].gt(df['yardline_100']),'left', 'right')

        # Adjust starting y coordinate because the perspective would change depending on the side.
        df['y_starting_dir'] = np.where(df['off_dir'] == 'right', df['y_starting'], df['y_starting'].rsub(53.3))

        #plot_histogram(df, 'y_starting', 50, 'off', 'y_pos_orig')
        #plot_histogram(df, 'y_starting_dir', 50, 'off', 'y_pos_notnorm')

        # Find starting position of qb and convert to float.
        df['y_starting_qb'] = df.groupby(['gameId', 'playId']).apply(lambda x: np.repeat(53.3/2, x.shape[0])
        if x[x['position'] == 'QB'].shape[0] == 0 else np.repeat(x[x['position'] == 'QB']['y_starting_dir'].iloc[0], x.shape[0])).explode().values
        df['y_starting_qb'] = df['y_starting_qb'].astype(float)

        # Find side of qb that player is lined up on.
        df['qb_side'] = np.where(df['y_starting_dir'].gt(df['y_starting_qb']), 'R', 'L')

        # Find the starting position of each player relative to the qb.
        df['y_starting_qb_dir'] = df['y_starting_dir'].sub(df['y_starting_qb'])

        # Find the order of positions based on offensive direction.
        # First, group and extract first value of the y starting position and direction.
        start_df = (df.groupby(['gameId', 'playId', 'position', 'nflId'])[['y_starting_dir', 'off_dir', 'qb_side']].first().reset_index())

        # Next, group and extract ranking of positions based on whether team is home or away
        # and the starting position.

        qb_start = start_df[start_df['position'] == 'QB']
        non_qb_start = start_df[start_df['position'] != 'QB']
        left = non_qb_start[non_qb_start['qb_side'] == 'L']
        right = non_qb_start[non_qb_start['qb_side'] == 'R']
        l = left.sort_values('y_starting_dir',ascending=False).groupby(['gameId', 'playId', 'position']).apply(
            lambda x: list(zip(x['nflId'], range(x.shape[1])))).explode().reset_index().rename({0: 'nfl_num'}, axis=1)
        r = right.sort_values('y_starting_dir').groupby(['gameId', 'playId', 'position']).apply(
            lambda x: list(zip(x['nflId'], range(x.shape[1])))).explode().reset_index().rename({0: 'nfl_num'}, axis=1)
        qb = qb_start.sort_values('y_starting_dir').groupby(['gameId', 'playId', 'position']).apply(
            lambda x: list(zip(x['nflId'], range(x.shape[1])))).explode().reset_index().rename({0: 'nfl_num'}, axis=1)
        l['qb_side'] = 'L'
        r['qb_side'] = 'R'
        full = pd.concat([l, r], axis=0)
        full['nflId'] = full['nfl_num'].map(lambda x: x[0])
        full['pos_order'] = full['nfl_num'].map(lambda x: x[1])
        full.drop('nfl_num', axis=1, inplace=True)

        qb['nflId'] = qb['nfl_num'].map(lambda x: x[0])
        qb['pos_order'] = qb['nfl_num'].map(lambda x: x[1])
        qb.drop('nfl_num', axis=1, inplace=True)
        qb_full = qb.merge(start_df, on=['gameId', 'playId', 'nflId', 'position'])
        full_w_qb = pd.concat([full, qb_full[['gameId', 'playId', 'position',
                                              'qb_side', 'nflId', 'pos_order']]], axis=0)
        start_df_full = start_df.merge(full_w_qb, on=['gameId', 'playId', 'nflId', 'position', 'qb_side'])

        start_df_full['posId'] = np.where(start_df_full['position'] != 'QB',
                                          start_df_full['position'].add(start_df_full['qb_side']).add(start_df_full['pos_order'].astype(str)),
                                          start_df_full['position'].add(start_df_full['pos_order'].astype(str)))


        # Merge full dataframe with position number dataframe.
        df = df.merge(start_df_full[['gameId', 'playId', 'nflId', 'posId', 'pos_order']], on=['gameId', 'playId', 'nflId'])

        # Use regex to extract personnel from personnel column, and concatenate with main dataframe.
        df = pd.concat([df, df['personnelD'].str.extract('(?P<DL>\d+) DL, (?P<LB>\d+) LB, (?P<DB>\d+) DB')], axis=1)

        # Find the position of each player relative to the line of scrimmage.
        df['x_behind_line'] = np.where(df['off_dir'] == 'right',
                                       df['absoluteYardlineNumber'].sub(df['x']),
                                       df['x'].sub(df['absoluteYardlineNumber']))

        df['x_behind_line_starting'] = df.groupby(['gameId', 'playId', 'nflId'])['x_behind_line'].transform('first')
        #plot_histogram(df, 'x_behind_line', 50, 'off', 'x_behind_line')

        # Extract the yardline for first down and line of scrimmage based on the
        # direction that the teams are facing.
        df['yardline_first_dir'] = np.where(df['off_dir'] == 'right',
                                            df['yardline_first'].rsub(100),
                                            df['yardline_first'])
        df['yardline_100_dir'] = np.where(df['off_dir'] == 'right',
                                          df['yardline_100'].rsub(100),
                                          df['yardline_100'])

        # Add flag if a player has gone at least 1 yard past the line of scrimmage.
        df['exceeded_1yd'] = df.groupby(['gameId', 'playId', 'nflId'])['x_behind_line'].transform(lambda x:
                                                                                                  x.max() > 1)

        # Use whether player is on offense, whether the player is a QB or WR, and whether a player has
        # moved 1 yard beyond the line of scrimmage to determine if the player is a receiver.
        df['receiver'] = (df['off'] & (df['position'] != 'QB') & (df['exceeded_1yd'] | (df['position'] == 'WR')))

        # Save offensive and defensive numbered position lists.
        #off_pos = df[df['off']]['posId'].unique()
        #def_pos = df[~df['off']]['posId'].unique()

        # Adjust y to match direction of offense.
        df['y_dir'] = np.where(df['off_dir'] == 'right', df['y'].rsub(53.3), df['y'])

        #plot_histogram(df, 'y_dir', 50, 'off', 'y_dir_hist')

        # Define y coordinates as relative to QB.
        df['y_dir_qb'] = df['y_dir'].sub(df['y_starting_qb'])
        df['y_dir_qb_starting'] = df.groupby(['gameId', 'playId', 'nflId'])['y_dir_qb'].transform('first')

        Dplayers_to_remove_df = df[(df['position'].isin(['QB','RB','HB','FB','TE','WR'])) &
                                   (df['off'] == False)][['gameId','playId']].drop_duplicates()
        Dplayers_to_remove_df['ids'] = Dplayers_to_remove_df['gameId'].astype(str) + Dplayers_to_remove_df['playId'].astype(str)
        df['gamePlayId'] = df['gameId'].astype(str) + df['playId'].astype(str)
        df = df[~df['gamePlayId'].isin(Dplayers_to_remove_df['ids'].to_list())]

        # The distribution looks centered around 0, as would be expected given that the QB lines up in the center.
        #plot_histogram(df, 'y_dir_qb', 50, 'off', 'y_qb_dist')
        print('...transformed...')
        return df

    def reduce_time(self, week):
        # Cut time accumulated into 10 deciles for each play in order to reduce the space. Can adjust number of cuts.
        df = self.transform_directions(week=week)
        time_cuts = df[['gameId', 'playId', 'time_acc_s']].drop_duplicates().groupby(['gameId', 'playId']).agg(
            lambda x: np.nan if x.shape[0] < self.n_cuts else pd.cut(x, self.n_cuts, labels=range(1,
                                                            self.n_cuts + 1))).explode('time_acc_s')
        time_cuts_idx = df[['gameId', 'playId', 'time_acc_s']].drop_duplicates().dropna()
        # print(len(time_cuts_idx))
        time_cuts_idx = time_cuts_idx[
            time_cuts_idx.groupby(['gameId', 'playId'])['time_acc_s'].transform(lambda x: x.shape[0] >= self.n_cuts)]
        # print(len(time_cuts))
        # print(len(time_cuts_idx))
        time_cuts_idx['time_cut'] = time_cuts.values

        df = df.merge(time_cuts_idx, on=['gameId', 'playId', 'time_acc_s'])

        # Aggregate by cur.
        full_cut_df = df.groupby(['gameId', 'playId', 'posId', 'time_cut']).agg(
            {'y_dir_qb': 'mean', 'x_behind_line': 'mean', 'off': 'first'}).reset_index()

        # Find offense and defence and merge.
        off_cut_df = full_cut_df[full_cut_df['off']]
        def_cut_df = full_cut_df[~full_cut_df['off']]
        cut_df = def_cut_df.merge(off_cut_df, on=['gameId', 'playId', 'time_cut'], suffixes=('_def', '_off'))
        cut_df['distance'] = np.linalg.norm(cut_df[['y_dir_qb_def', 'x_behind_line_def']].values -
                                            cut_df[['y_dir_qb_off', 'x_behind_line_off']].values, axis=1)

        # Find distance to each offensive player and use to find closest player.
        cut_df['dist_min'] = cut_df.groupby(['gameId', 'playId', 'posId_def', 'time_cut'])['distance'].transform('min')
        print('...time reduced...')
        return df, cut_df

    def simple_closest_player(self, df, cut_df):
        cut_df = cut_df[cut_df['distance'] == cut_df['dist_min']]
        cut_df = (cut_df[['gameId', 'playId', 'time_cut', 'posId_def', 'posId_off']]
                  .rename({'posId_def': 'posId', 'posId_off': 'pos_off_closest'}, axis=1))
        df = df.merge(cut_df, on=['gameId', 'playId', 'time_cut', 'posId'], how='left')

        # Next, determine minimum distances between each defensive player and receiver and qb.
        # Separate defensive and receiver dataframes.
        rec_dis_df = df[df['receiver']]
        def_dis_df = df[~df['off']]

        # Merge defensive with receiver dataframes on game, play, and time.
        dis_df = def_dis_df.merge(rec_dis_df[['gameId', 'playId', 'time_acc_s', 'x_behind_line', 'y_dir_qb']],
                                  on=['gameId', 'playId', 'time_acc_s'],
                                  suffixes=['_def', '_rec'])


        # Find distance between each defensive player and each receiver.
        dis_df['dist'] = np.linalg.norm(dis_df[['x_behind_line_def', 'y_dir_qb_def']].values -
                                        dis_df[['x_behind_line_rec', 'y_dir_qb_rec']].values, axis=1)

        # Group dataframe to obtain minimum distance.
        min_dis_df = dis_df.groupby(['gameId', 'playId', 'time_acc_s', 'posId'])['dist'].min()
        min_dis_df.name = 'min_dist_rec'

        # Separate QB dataframe
        qb_df = df[df['position'] == 'QB']

        # Merge defensive with QB dataframe.
        qb_df = def_dis_df.merge(qb_df[['gameId', 'playId', 'time_acc_s', 'x_behind_line', 'y_dir_qb']],
                                 on=['gameId', 'playId', 'time_acc_s'],
                                 suffixes=['_def', '_qb'])

        # Find distance to the QB.
        qb_df['dist'] = np.linalg.norm(qb_df[['x_behind_line_def', 'y_dir_qb_def']].values -
                                       qb_df[['x_behind_line_qb', 'y_dir_qb_qb']].values, axis=1)

        # Group to form index and distance.
        qb_df = qb_df.groupby(['gameId', 'playId', 'time_acc_s', 'posId'])['dist'].min()
        qb_df.name = 'dist_qb'

        # Concatenate about the same index and reset the index.
        min_dist = pd.concat([min_dis_df, qb_df], axis=1).reset_index()

        # Merge main dataframe with minimum distance dataframe.
        df = df.merge(min_dist, on=['gameId', 'playId', 'time_acc_s', 'posId'], how='left')

        # Evaluate whether a receiver is closer than the qb.
        df['rec_closer'] = df['min_dist_rec'].lt(df['dist_qb'])
        print('...distance calculated...')

        return df

    def cosine_closest_player(self, cut_df, n_closest=3):
        top_closest_players = (cut_df[cut_df['distance'] == cut_df['dist_min']]
            .groupby(['gameId', 'playId', 'posId_def', 'time_cut'])['posId_off'].first()
            .reset_index()
            .rename({'posId_off': 'posId_off_closest'}, axis=1)
                               )
        closest_players = cut_df.sort_values('distance').groupby(
            ['gameId', 'playId', 'posId_def', 'time_cut']).head(n_closest)
        cut_even = closest_players[closest_players['time_cut'].map(lambda x: x % 2 == 0)]
        cut_odd = closest_players[closest_players['time_cut'].map(lambda x: x % 2 == 1)]
        cut_even['prev_cut'] = cut_even['time_cut'].sub(1)
        cuts_even_merged = cut_even.merge(cut_odd,
                                          left_on=['gameId', 'playId', 'posId_def', 'posId_off', 'prev_cut'],
                                          right_on=['gameId', 'playId', 'posId_def', 'posId_off', 'time_cut'],
                                          suffixes=('_time2', '_time1'))
        cut_odd['prev_cut'] = cut_odd['time_cut'].sub(1)
        cut_even.drop('prev_cut', axis=1, inplace=True)
        cut_odd = cut_odd[cut_odd['prev_cut'].gt(0)]
        cuts_odd_merged = cut_odd.merge(cut_even,
                                        left_on=['gameId', 'playId', 'posId_def', 'posId_off', 'prev_cut'],
                                        right_on=['gameId', 'playId', 'posId_def', 'posId_off', 'time_cut'],
                                        suffixes=('_time2', '_time1'))
        full_cut_merged = pd.concat([cuts_even_merged, cuts_odd_merged], axis=0)
        full_cut_merged = full_cut_merged.sort_values(
            ['gameId', 'playId', 'posId_def', 'time_cut_time1', 'posId_off'])
        full_cut_merged['y_dir_qb_def_delta'] = full_cut_merged['y_dir_qb_def_time2'].sub(
            full_cut_merged['y_dir_qb_def_time1'])
        full_cut_merged['x_behind_line_def_delta'] = full_cut_merged['x_behind_line_def_time2'].sub(
            full_cut_merged['x_behind_line_def_time1'])
        def_vector = np.concatenate([
            full_cut_merged['x_behind_line_def_delta'].values.reshape(-1, 1),
            full_cut_merged['y_dir_qb_def_delta'].values.reshape(-1, 1)], axis=1)
        full_cut_merged['y_dir_qb_off_delta'] = full_cut_merged['y_dir_qb_off_time2'].sub(
            full_cut_merged['y_dir_qb_off_time1'])
        full_cut_merged['x_behind_line_off_delta'] = full_cut_merged['x_behind_line_off_time2'].sub(
            full_cut_merged['x_behind_line_off_time1'])
        off_vector = np.concatenate([
            full_cut_merged['x_behind_line_off_delta'].values.reshape(-1, 1),
            full_cut_merged['y_dir_qb_off_delta'].values.reshape(-1, 1)], axis=1)
        cosine_sim = np.sum(def_vector * off_vector, axis=1)/(
                np.linalg.norm(def_vector, axis=1) * np.linalg.norm(off_vector, axis=1))
        full_cut_merged['cosine_sim'] = cosine_sim
        max_cosine = full_cut_merged.groupby(['gameId', 'playId', 'posId_def', 'time_cut_time2'])['cosine_sim'].agg(
            lambda x: np.argmax(x)
        )
        max_cosine.name = 'max_cosine_idx'

        full_cut_cosine = full_cut_merged.merge(max_cosine,
                                                left_on=['gameId', 'playId', 'posId_def',
                                                         'time_cut_time2'],
                                                right_index=True)
        min_cosine = full_cut_merged.groupby(['gameId', 'playId', 'posId_def', 'time_cut_time2'])['cosine_sim'].agg(
            lambda x: np.argmin(x)
        )
        min_cosine.name = 'min_cosine_idx'
        full_cut_cosine_max_min = full_cut_cosine.merge(min_cosine,
                                                        left_on=['gameId', 'playId', 'posId_def',
                                                                             'time_cut_time2'],
                                                        right_index=True)
        full_cut_cosine_max_min['player_idx'] = full_cut_cosine_max_min.groupby(
            ['gameId', 'playId', 'posId_def', 'time_cut_time2']).apply(
            lambda x: range(x.shape[0])).explode().values
        full_top_cosine = full_cut_cosine_max_min[full_cut_cosine_max_min['max_cosine_idx'] ==
                                                  full_cut_cosine_max_min['player_idx']]
        full_top_cosine.rename({'posId_off': 'posId_off_max'}, axis=1, inplace=True)
        full_bot_cosine = full_cut_cosine_max_min[full_cut_cosine_max_min['min_cosine_idx'] ==
                                                  full_cut_cosine_max_min['player_idx']]
        full_bot_cosine.rename({'posId_off': 'posId_off_min'}, axis=1, inplace=True)
        player_closest_cosine = full_top_cosine.merge(full_bot_cosine,
                                                      on=['gameId', 'playId',
                                                          'posId_def', 'time_cut_time2'])
        full_cosine = player_closest_cosine[
            ['gameId', 'playId', 'posId_def', 'time_cut_time2', 'posId_off_max', 'posId_off_min']]
        full_cosine_closest = full_cosine.merge(top_closest_players.rename({
            'time_cut': 'time_cut_time2'
            }, axis=1), on=['gameId', 'playId', 'posId_def', 'time_cut_time2'])
        print('...closest player based on cosine similarity calculated...')
        return full_cosine_closest

    def starting_pos(self, full_df):

        trans_df = full_df[['gameId', 'playId', 'posId', 'y_dir_qb_starting', 'x_behind_line_starting',
                            'defendersInTheBox','numberOfPassRushers', 'DB', 'LB', 'DL', 'off',
                            'yardline_100', 'yardline_first']].drop_duplicates()
        trans_df_def = trans_df[~trans_df['off']]
        trans_df_def.drop('off', axis=1, inplace=True)
        trans_stacked = (trans_df_def.set_index(['gameId', 'playId', 'posId',
                                                 'defendersInTheBox','numberOfPassRushers','DB', 'LB', 'DL',
                                                 'yardline_first', 'yardline_100'])
                         .stack()
                         .reset_index()
                         .rename({'level_10': 'starting', 0: 'value'}, axis=1)
                         .replace({'y_dir_qb_starting': 'y_start', 'x_behind_line_starting': 'x_start'})
                         )
        trans_stacked['posId'] = trans_stacked['posId'].add('_').add(trans_stacked['starting'])
        trans_stacked.drop('starting', axis=1, inplace=True)
        start_df = trans_stacked[['gameId', 'playId', 'posId', 'value']]
        info_df = trans_stacked[['gameId', 'playId', 'defendersInTheBox','numberOfPassRushers', 'DB', 'LB', 'DL',
                                 'yardline_first', 'yardline_100']].drop_duplicates()
        print('...starting dataframe generated...')
        return start_df, info_df

    def generate_action_type_df(self, full_df, cut_df):
        cut_df = cut_df[cut_df['distance'] == cut_df['dist_min']]
        cut_df = (cut_df[['gameId', 'playId', 'time_cut', 'posId_def', 'posId_off']]
                  .rename({'posId_def': 'posId', 'posId_off': 'pos_off_closest'}, axis=1))
        df = full_df.merge(cut_df, on=['gameId', 'playId', 'time_cut', 'posId'], how='left')
        # Aggregate columns based on game, play, numbered position, and time quartile.
        df = df[~df['off'] & (df['position'] != 'TE')].groupby(['gameId', 'playId', 'posId', 'time_cut']).agg(
            {'pos_off_closest': 'first'}
        ).reset_index()

        action_df = df[['gameId','playId','posId','time_cut','pos_off_closest']]

        action_group_df = action_df.groupby(by=['gameId','playId','posId','pos_off_closest']).mean().reset_index().set_index(['gameId','playId','posId'])

        pos_df = action_group_df.reset_index()
        actions = np.where((pos_df.groupby(['gameId', 'playId'])['posId'].transform(lambda x: x.shape[0] == 1)
                            & pos_df['pos_off_closest'].str.contains('QB')) | pos_df.sort_values(
            ['time_cut', 'pos_off_closest'], ascending=[True, False]
        )
                           .groupby(['gameId', 'playId', 'posId'])['pos_off_closest'].transform(
            lambda x: x.iloc[-1][:2] == 'QB'), 'B',
                           np.where(pos_df.groupby(['gameId', 'playId', 'posId'])['pos_off_closest']
                                    .transform(lambda x: (x.map(lambda x: x[:2]) != 'QB').sum() == 1),
                                    'M', 'Z'))
        action_group_df['def_action'] = actions
        action_group_df.drop(['pos_off_closest', 'time_cut'], axis=1, inplace=True)
        result_df = action_group_df.reset_index().drop_duplicates()
        result_df['posId'] = result_df['posId'].add('_act')
        result_df.rename({'def_action': 'value'}, axis=1, inplace=True)
        print('...action type generated...')
        return result_df

    def combine_week(self, week):
        full_df, cut_df = self.reduce_time(week)
        start_df, info_df = self.starting_pos(full_df)
        if self.simMethod == 'distance':
            action_df = self.generate_action_type_df(full_df, cut_df)
        else:
            raise InvalidSimilarityMetricError
        total_df =  pd.concat([action_df,start_df],axis=0)
        total_df_info = total_df.merge(info_df, on=['gameId', 'playId'])
        total_df_info['week'] = week
        print('.....Week {} COMPLETE.....'.format(str(week)))
        return total_df_info

    def generate_full_df(self, first, last, fp='def_clean_output.csv'):
        output_df = pd.DataFrame()
        start_time = time.time()
        for week in range(first, last+1):
            total_df = self.combine_week(week=week)
            output_df = pd.concat([output_df, total_df], axis=0)
            print('')
            percent_complete = round((week - first + 1) / (last - first + 1)*100,2)
            print('   {}% COMPLETE   '.format(str(percent_complete)))
            print('')
            end_time = time.time()
            print("--- {} minutes elapsed ---".format(round((end_time - start_time)/60,1)))
            print('')
            print("the weeks complete: ", output_df.week.unique())
        output_df = output_df.pivot(index=['gameId', 'playId', 'defendersInTheBox','numberOfPassRushers', 'DB', 'LB', 'DL',
                                           'yardline_first', 'yardline_100'], columns='posId',values='value')
        output_df.to_csv(f'assets/{fp}')
        print(f"Defensive cleaning complete --- check assets/{fp}")
        return output_df