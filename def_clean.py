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
        if not os.path.exists('assets/full_position.csv'):
            self.weeks_data = get_positional_data()
        else:
            self.weeks_data = pd.read_csv('assets/full_position.csv')
        self.play_data = pd.read_csv('Kaggle-Data-Files/plays.csv')
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

    def find_rank(self, df, col, reverse=False):
        """
        Find the ranking of a series based on values.
        :param df: Dataframe for ranking; pd.DataFrame
        :param col: Column from dataframe to rank; str
        :param reverse: Flag of whether to reverse rank direction; bool
        :return: Array with rankings; np.array
        """
        # Extract series and use arsort to find rankings.
        ser = df[col]
        temp = np.argsort(ser)

        # Reverse direction based on flag.
        if reverse:
            temp = temp[::-1]

        # Fill ranking array.
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(ser.shape[0])
        return ranks

    def plot_histogram(self, df, column, bins, hue, fig_name):
        sns.histplot(data=df, x=column, bins=bins, hue=hue)
        figure = 'assets/' + fig_name + '.png'
        plt.savefig(figure)
        plt.close()

    def transform_directions(self, week):
        # Extract the time elapsed in the play. Labeled as "time_acc_s". May take a while for full dataset.
        df = self.filter_full_position_df(week=week)
        play_data = self.play_data.copy()
        df['time_diff'] = df.groupby(['playId', 'gameId', 'displayName'])['time'].diff()
        df['time_diff'][df['time_diff'].isnull()] = pd.Timedelta(0)
        df['time_acc_s'] = df.groupby(['playId', 'gameId', 'displayName'])['time_diff'].transform(
            lambda x: x.map(lambda x: x.microseconds).cumsum()).div(1e6)


        play_df = play_data[play_data['absoluteYardlineNumber'].notnull()]
        #print(play_df.shape)
        play_df = play_df[['gameId','playId', 'absoluteYardlineNumber','yardsToGo','personnelD']]

        # Merge movement and play-by-play datasets.
        df = df.merge(play_df, on=['gameId', 'playId'])

        #print(df.columns)

        # Find which teams in the dataframe are offensive vs. defensive.
        df['off'] = np.where(df['position'].isin(['QB', 'HB', 'FB', 'WR', 'TE', 'C', 'OG', 'OT', 'RB']),
                             True, False)
        #print("check for any offensive positions not mapped", df[~df['off']]['position'].unique())

        # Extract starting x and y position.
        df['x_starting'] = df.groupby(['gameId', 'playId', 'nflId'])['x'].transform(lambda x: x.iloc[0])
        df['y_starting'] = df.groupby(['gameId', 'playId', 'nflId'])['y'].transform(lambda x: x.iloc[0])

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
        df['y_starting_dir'] = np.where(df['off_dir'] == 'right', df['y_starting'].rsub(53.3), df['y_starting'])

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
        order_col = np.where(start_df['position'] != 'QB',
                             (start_df.groupby(['gameId', 'playId', 'position', 'qb_side'])
                              .apply(lambda x: np.where(x.index.get_level_values(-1) == 'R',
                                                        self.find_rank(x, 'y_starting_dir'),
                                                        self.find_rank(x, 'y_starting_dir', reverse=True)))
                              .explode()
                              .values
                              ),
                             (start_df.groupby(['gameId', 'playId', 'position'])
                              .apply(lambda x: self.find_rank(x, 'y_starting_dir'))
                              .explode()
                              .values
                              )
                             )
        # Add column with the position order to the df with indexed starting position.
        start_df['pos_order'] = order_col

        # Concatenate position and position order to create unique position identifier.
        start_df['posId'] = np.where(start_df['position'] != 'QB',
                                     start_df['position'].add(start_df['qb_side']).add(start_df['pos_order'].astype(str)),
                                     start_df['position'].add(start_df['pos_order'].astype(str)))


        # Merge full dataframe with position number dataframe.
        df = df.merge(start_df[['gameId', 'playId', 'nflId', 'posId', 'pos_order']], on=['gameId', 'playId', 'nflId'])

        # Use regex to extract personnel from personnel column, and concatenate with main dataframe.
        df = pd.concat([df, df['personnelD'].str.extract('(?P<DL>\d+) DL, (?P<LB>\d+) LB, (?P<DB>\d+) DB')], axis=1)

        # Find the position of each player relative to the line of scrimmage.
        df['x_behind_line'] = np.where(df['off_dir'] == 'right',
                                       df['absoluteYardlineNumber'].sub(df['x']),
                                       df['x'].sub(df['absoluteYardlineNumber']))


        #plot_histogram(df, 'x_behind_line', 50, 'off', 'x_behind_line')

        # Extract the yardline for first down and line of scrimmage based on the
        # direction that the teams are facing.
        df['yardline_first_dir'] = np.where(df['off_dir'] == 'right',
                                            df['yardline_first'],
                                            df['yardline_first'].rsub(100))
        df['yardline_100_dir'] = np.where(df['off_dir'] == 'right',
                                          df['yardline_100'],
                                          df['yardline_100'].rsub(100))

        # Add flag if a player has gone at least 1 yard past the line of scrimmage.
        df['exceeded_1yd'] = df.groupby(['gameId', 'playId', 'nflId'])['x_behind_line'].transform(lambda x: x.max() > 1)

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
        full_cut_df = df.groupby(['gameId', 'playId', 'posId', 'time_cut']).agg({'y_dir_qb': 'mean', 'x_behind_line': 'mean', 'off': 'first'}).reset_index()

        # Find offense and defence and merge.
        off_cut_df = full_cut_df[full_cut_df['off']]
        def_cut_df = full_cut_df[~full_cut_df['off']]
        cut_df = def_cut_df.merge(off_cut_df, on=['gameId', 'playId', 'time_cut'], suffixes=('_def', '_off'))
        print('...time reduced...')
        return df, cut_df

    def simple_closest_player(self, df, cut_df):

        # Find distance to each offensive player and use to find closest player.
        cut_df['distance'] = np.linalg.norm(cut_df[['y_dir_qb_def', 'x_behind_line_def']].values -
                                            cut_df[['y_dir_qb_off', 'x_behind_line_off']].values, axis=1)
        cut_df['dist_min'] = cut_df.groupby(['gameId', 'playId', 'posId_def', 'time_cut'])['distance'].transform('min')

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

    def starting_pos(self, full_df):

        full_df=full_df[['gameId','playId','posId','x_starting','y_starting','off']].drop_duplicates()

        xy_df = full_df[~full_df['off']].drop(columns='off')

        x_df = xy_df[['gameId','playId','posId','x_starting']].drop_duplicates()

        y_df = xy_df[['gameId','playId','posId','y_starting']].drop_duplicates()

        y_df = y_df.groupby(['gameId','playId','posId']).mean().reset_index().rename(columns={'y_starting':'value'})
        y_df['posId'] = y_df['posId'] + '_y_start'

        x_df = x_df.groupby(['gameId','playId','posId']).mean().reset_index().rename(columns={'x_starting':'value'})
        x_df['posId'] = x_df['posId'] + '_x_start'

        start_df = pd.concat([x_df,y_df],axis=0)
        print('...starting dataframe generated...')
        return start_df

    def return_action_by_pos(self, df):
        play_pos_Ids = df.index

        action_types = []

        for ppId in play_pos_Ids:

            pos = df.loc[ppId]

            pos_no_qb = pos[pos['pos_off_closest']!='QB0']

            action = None

            if len(pos) == 1 and pos['pos_off_closest'].values == 'QB0':
                action = "B"

            if len(pos_no_qb) == 1:
                action = "M"
            else:
                action = "Z"

            if pos[pos.time_cut == pos.time_cut.max()]['pos_off_closest'].values[0] == 'QB0':
                action = "B"

            action_types.append([ppId[0],ppId[1],ppId[2], action])

        return pd.DataFrame(action_types,
                            columns=['gameId','playId','posId','def_action']).set_index(
            ['gameId','playId','posId'])

    def generate_action_type_df(self, df):
        # Aggregate columns based on game, play, numbered position, and time quartile.
        df = df[~df['off'] & (df['position'] != 'TE')].groupby(['gameId', 'playId', 'posId', 'time_cut']).agg(
            {'x_starting': 'first', 'y_starting': 'first',
             'yardline_100_dir': 'first', 'yardline_first_dir': 'first',
             'DL': 'first', 'LB': 'first', 'DB': 'first',
             'x_behind_line': 'mean', 'y_dir': 'mean', 'min_dist_rec': 'mean',
             'dist_qb': 'mean',
             'pos_off_closest': 'first'}
        ).reset_index()

        action_df = df[['gameId','playId','posId','time_cut','pos_off_closest']]

        action_group_df = action_df.groupby(by=['gameId','playId','posId','pos_off_closest']).mean().reset_index().set_index(['gameId','playId','posId'])

        #pos = action_group_df.loc[(2018090600,   889,  'MLBR0')]

        #print(pos[pos.time_cut == pos.time_cut.max()]['pos_off_closest'].values[0])

        result_df = self.return_action_by_pos(action_group_df)

        result_df = result_df[~result_df.index.duplicated(keep='first')]

        review_df = action_df.merge(result_df, on=['gameId', 'playId', 'posId'], how='left')

        #complete_df['key'] = complete_df['gameId'].astype(str) + complete_df['playId'].astype(str) + complete_df['posId']
        review_df.to_csv('assets/action_type_over_time_cuts.csv')
        #complete_df.reset_index().pivot(index=['gameId', 'playId'], columns='posId',values='def_action')

        #result_df = result_df.reset_index().pivot(index=['gameId', 'playId'], columns='posId',values='def_action')
        result_df = result_df.reset_index().rename(columns={'def_action':'value'})

        result_df['posId'] = result_df['posId'] + '_act'
        print('...action type dataframe generated...')
        return result_df

    def combine_week(self, week):
        full_df, cut_df = self.reduce_time(week)
        start_df = self.starting_pos(full_df)
        if self.simMethod == 'distance':
            simple_df = self.simple_closest_player(full_df, cut_df)
            action_df = self.generate_action_type_df(simple_df)
        else:
            raise InvalidSimilarityMetricError
        total_df =  pd.concat([action_df,start_df],axis=0)
        total_df['week'] = week
        print('.....Week {} COMPLETE.....'.format(str(week)))
        return total_df

    def generate_full_df(self, first, last):
        output_df = pd.DataFrame()
        start_time = time.time()
        for week in range(first, last+1):
            total_df = self.combine_week(week=week)
            output_df = pd.concat([output_df, total_df], axis=0)
            print('')
            percent_complete = round(week / (last - first)*100,2)
            print('   {}% COMPLETE   '.format(str(percent_complete)))
            print('')
            end_time = time.time()
            print("--- {} minutes elapsed ---".format(round((end_time - start_time)/60,1)))
            print('')
            print("the weeks complete: ", output_df.week.unique())
        output_df = output_df.pivot(index=['gameId', 'playId'], columns='posId',values='value')
        output_df.to_csv('assets/def_clean_output.csv')
        print("Defensive cleaning complete --- check assets/def_clean_output.csv")
        return output_df