import pandas as pd
import numpy as np
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
    """
    Prepares the defensive positional data for downstream machine learning tasks

    Attributes
    ----------
    weeks_data : pandas DataFrame
        Flag for whether DataFrame is loaded model is specified, or specified model.
        default : None
    n_cuts : int
        Number of cuts to reduce the chunks of time into
        default : 11
    frameLimit : int
        Number of frames required in a play to use in downstream tasks
        default : 11
    simMethod: str
        Method to calculate the closest offensive player
        default : "distance"

    Methods
    -------
    filter_full_position_df
        Filter plays in positional data for time data between ball snap and pass forward greater than frameLimit
    transform_directions
        Transform the player's movement data to normalize offense direction, extract relevant 
        football distance metrics and label defensive player based on QB perspective
    reduce_time
        Reduce the play data into the number of cuts 'n_cuts' and calculate closest offensive player 
        for each defender across the cuts
    starting_pos
        Generate starting position for each defender and extract play-level information
    generated_action_type_df
        Generate the action type (blitz, zone or man) for each defender
    combine_week
        Utilize reduce_time, starting_pos and generate_action_type_df to create dataframe for a single week
    generate_full_df
        Run combine_week for consecutive number of weeks specified and output to a CSV file
    """
    def __init__(self, weeks_data=None, n_cuts=11, frameLimit=11, simMethod='distance'):
        print('..............................initializing')
        if not os.path.exists('Kaggle-Data-Files'):
            get_assets()
        if type(weeks_data) != pd.DataFrame:
            self.weeks_data = get_positional_data()
        else:
            self.weeks_data = weeks_data
        self.play_data = pd.read_csv('Kaggle-Data-Files/plays.csv')
        self.game_data = pd.read_csv('Kaggle-Data-Files/games.csv')
        print('..data downloaded...')
        self.n_cuts = n_cuts
        self.frameLimit = frameLimit
        self.simMethod = simMethod

    def filter_full_position_df(self, week):
        """
        Filter plays in positional data for time data between ball snap and pass forward greater than frameLimit

        Parameters
        ----------
        week : int
            Specified week to filter data for

        Returns
        -------
        Filtered dataFrame for specified week
        """
        # Filter raw week data for a specified week.
        df = self.weeks_data[self.weeks_data['week'] == week]
        print('...Week {} loaded...'.format(str(week)))
        
        # Split df into a football only and position only DataFrames.
        fb_df = df[df['nflId'].isnull()]
        pos_df = df[df['nflId'].notnull()]

        # Convert time to datetime format.
        fb_df['time'] =  pd.to_datetime(fb_df['time'], format='%Y-%m-%dT%H:%M:%S')

        # Find time that pass was thrown and merge with main df.
        pass_start = fb_df[fb_df['event'] == 'pass_forward'][['gameId', 'playId', 'frameId']].rename({'frameId': 'frame_pass'}, axis=1)

        # Find time that ball was snapped and merge with main df
        ball_snap = fb_df[fb_df['event'] == 'ball_snap'][['gameId', 'playId', 'frameId']].rename({'frameId': 'frame_snap'}, axis=1)

        # Merge pass start and ball snap to the position df
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

        # Create dataframe that counts number of frames in a play
        uniq_df = pos_df[['frameId','gameId','playId','event']]
        uniq_df = uniq_df.groupby(by=['frameId','gameId','playId']).count().reset_index().drop(columns='event')
        uniq_df = uniq_df.groupby(by=['gameId','playId']).count().sort_values(by='frameId').rename(columns={'frameId':'frameCount'})

        # Merge frameCount to positional df
        pos_df = pos_df.merge(uniq_df.reset_index(), on=['gameId','playId'])

        # Filter positional df for any plays greater than or equal to frameLimit
        pos_df = pos_df[pos_df['frameCount'] >= self.frameLimit]

        print('...filtered...')
        return pos_df.drop(columns=['before_pass','after_snap'])

    def transform_directions(self, week):
        """
        Transform the player's movement data to normalize offense direction, extract relevant 
        football distance metrics and label defensive player based on QB perspective

        Parameters
        ----------
        week : int
            Specified week to filter data for

        Returns
        -------
        Transformed dataFrame for specified week
        """
        # Filter df for specified week
        df = self.filter_full_position_df(week=week)
        
        # Copy play_data 
        play_data = self.play_data.copy()
        
        # Calcualte the time accumulated in seconds across players, plays and games
        df['time_diff'] = df.groupby(['playId', 'gameId', 'displayName'])['time'].diff()
        df['time_diff'][df['time_diff'].isnull()] = pd.Timedelta(0)
        df['time_acc_s'] = df['time_diff'].dt.microseconds.div(1e6)
        df['time_acc_s'] = df.groupby(['playId', 'gameId', 'nflId'])['time_acc_s'].transform("cumsum")
        df.drop('time_diff', axis=1, inplace=True)

        # Remove records from with nulls in absoluteYardlineNumber
        play_df = play_data[play_data['absoluteYardlineNumber'].notnull()]
    
        # Remove unecessary columns from play_df
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

        
        # Drop staging columns to determine offense or defense
        df.drop(['possessionTeam', 'visitorTeamAbbr', 'homeTeamAbbr'], axis=1, inplace=True)

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
        # and the starting position/features including which side of the QB the player is on.
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

        # Adjust y to match direction of offense.
        df['y_dir'] = np.where(df['off_dir'] == 'right', df['y'].rsub(53.3), df['y'])

        # Define y coordinates as relative to QB.
        df['y_dir_qb'] = df['y_dir'].sub(df['y_starting_qb'])
        df['y_dir_qb_starting'] = df.groupby(['gameId', 'playId', 'nflId'])['y_dir_qb'].transform('first')

        # Remove defensive player that are classified on offense
        Dplayers_to_remove_df = df[(df['position'].isin(['QB','RB','HB','FB','TE','WR'])) &
                                   (df['off'] == False)][['gameId','playId']].drop_duplicates()
        Dplayers_to_remove_df['ids'] = Dplayers_to_remove_df['gameId'].astype(str) + Dplayers_to_remove_df['playId'].astype(str)
        df['gamePlayId'] = df['gameId'].astype(str) + df['playId'].astype(str)
        df = df[~df['gamePlayId'].isin(Dplayers_to_remove_df['ids'].to_list())]

        print('...transformed...')
        return df

    def reduce_time(self, week):
        """
        Reduce the play data into the number of cuts 'n_cuts' and calculate closest offensive player 
        for each defender across the cuts    

        Parameters
        ----------
        week : int
            Specified week to filter data for

        Returns
        -------
        Full dataFrame & cut dataFrame containing closest offesnive player for specified week
        """
        # Return transformed dataframe for specified week.
        df = self.transform_directions(week=week)
        
        # Cut time accumulated into 11 deciles for each play in order to reduce the space.
        time_cuts = df[['gameId', 'playId', 'time_acc_s']].drop_duplicates().groupby(['gameId', 'playId']).agg(
            lambda x: np.nan if x.shape[0] < self.n_cuts else pd.cut(x, self.n_cuts, labels=range(1,
                                                            self.n_cuts + 1))).explode('time_acc_s')
        
        
        # Create index across time stamps.
        time_cuts_idx = df[['gameId', 'playId', 'time_acc_s']].drop_duplicates().dropna()

        # Identify time cuts for n_cuts across the duration of the play and merge with original DataFrame.
        time_cuts_idx = time_cuts_idx[
            time_cuts_idx.groupby(['gameId', 'playId'])['time_acc_s'].transform(lambda x: x.shape[0] >= self.n_cuts)]
        time_cuts_idx['time_cut'] = time_cuts.values
        df = df.merge(time_cuts_idx, on=['gameId', 'playId', 'time_acc_s'])

        # Aggregate by cut.
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

    def starting_pos(self, full_df):
        """
        Generate starting position for each defender and extract play-level information

        Parameters
        ----------
        full_df : pd.DataFrame
            Data that has not been reduced to specified number of cuts

        Returns
        -------
        DataFrame containing defender's starting positions and DataFrame containing defensive info
        """
        # remove any offensive players from the full_df
        trans_df = full_df[['gameId', 'playId', 'posId', 'y_dir_qb_starting', 'x_behind_line_starting',
                            'defendersInTheBox','numberOfPassRushers', 'DB', 'LB', 'DL', 'off',
                            'yardline_100_dir', 'yardline_first_dir']].drop_duplicates()
        trans_df_def = trans_df[~trans_df['off']]
        trans_df_def.drop('off', axis=1, inplace=True)
        
        # identify starting position for each defensive player in a long table
        trans_stacked = (trans_df_def.set_index(['gameId', 'playId', 'posId',
                                                 'defendersInTheBox','numberOfPassRushers','DB', 'LB', 'DL',
                                                 'yardline_first_dir', 'yardline_100_dir'])
                         .stack()
                         .reset_index()
                         .rename({'level_10': 'starting', 0: 'value'}, axis=1)
                         .replace({'y_dir_qb_starting': 'y_start', 'x_behind_line_starting': 'x_start'})
                         )
        trans_stacked['posId'] = trans_stacked['posId'].add('_').add(trans_stacked['starting'])
        trans_stacked.drop('starting', axis=1, inplace=True)
        
        # create starting position dataframe
        start_df = trans_stacked[['gameId', 'playId', 'posId', 'value']]
        
        # create dataframe containing play metadata
        info_df = trans_stacked[['gameId', 'playId', 'defendersInTheBox','numberOfPassRushers', 'DB', 'LB', 'DL',
                                 'yardline_first_dir', 'yardline_100_dir']].drop_duplicates()
        print('...starting dataframe generated...')
        return start_df, info_df

    def generate_action_type_df(self, full_df, cut_df):
        """
        Generate the action type (blitz, zone or man) for each defender

        Parameters
        ----------
        full_df : pd.DataFrame
            Data from reduce_time function containing play metadata 
        cut_df : pd.DataFrame
            Data from reduce_time function that has been reduce into n_cuts for each play

        Returns
        -------
        DataFrame containing action type codes for each defender across all plays
        """

        # Extract the defender and offensive player closet to defender, merge to original DataFrame
        cut_df = cut_df[cut_df['distance'] == cut_df['dist_min']]
        cut_df = (cut_df[['gameId', 'playId', 'time_cut', 'posId_def', 'posId_off']]
                  .rename({'posId_def': 'posId', 'posId_off': 'pos_off_closest'}, axis=1))
        df = full_df.merge(cut_df, on=['gameId', 'playId', 'time_cut', 'posId'], how='left')
        
        # Aggregate columns based on game, play, numbered position, and time quartile.
        df = df[~df['off'] & (df['position'] != 'TE')].groupby(['gameId', 'playId', 'posId', 'time_cut']).agg(
            {'pos_off_closest': 'first'}
        ).reset_index()

        # Remove uncessary columns for action generation in new df
        action_df = df[['gameId','playId','posId','time_cut','pos_off_closest']]

        # Aggregate the mean time_cut across defenders and closest offensive players
        action_group_df = action_df.groupby(by=['gameId','playId','posId','pos_off_closest']).mean().reset_index().set_index(['gameId','playId','posId'])

        # Simple algorithm to generate defenders action type
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
        
        # Take actions generated and create resulting DataFrame with defender's action
        action_group_df['def_action'] = actions
        action_group_df.drop(['pos_off_closest', 'time_cut'], axis=1, inplace=True)
        result_df = action_group_df.reset_index().drop_duplicates()
        result_df['posId'] = result_df['posId'].add('_act')
        result_df.rename({'def_action': 'value'}, axis=1, inplace=True)
        
        print('...action type generated...')
        return result_df

    def combine_week(self, week):
        """
        Utilize reduce_time, starting_pos and generate_action_type_df to create dataframe for a single week

        Parameters
        ----------
        week : int
            Week number

        Returns
        -------
        DataFrame of one processed week
        """
        # Reduce plays to specified time cuts, extract starting positions and generate action types
        full_df, cut_df = self.reduce_time(week)
        start_df, info_df = self.starting_pos(full_df)
        if self.simMethod == 'distance':
            action_df = self.generate_action_type_df(full_df, cut_df)
        else:
            raise InvalidSimilarityMetricError
        
        # Combine actions and starting positions into single DataFrame
        total_df =  pd.concat([action_df,start_df],axis=0)

        # Add in play metadata and add a column with week number
        total_df_info = total_df.merge(info_df, on=['gameId', 'playId'])
        total_df_info['week'] = week
        print('.....Week {} COMPLETE.....'.format(str(week)))
        return total_df_info

    def generate_full_df(self, first, last, fp='../assets/def_clean_output.csv'):
        """
        Run combine_week for consecutive number of weeks specified and output to a CSV file

        Parameters
        ----------
        first : int
            Starting week number to process
        last : int
            Ending week number to process
        fp : str
            File path of processed dataframe, include .csv at end
            default : def_clean_output.csv'

        Returns
        -------
        DataFrame of all processed weeks specified
        """
        # Create empty DataFrame and begin timing the processing
        output_df = pd.DataFrame()
        start_time = time.time()
        
        # Iterate through consecutive weeks, printing % complete and time elapsed as each week complete
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
        
        # Pivot data to include defender's actions and starting positions
        output_df = output_df.pivot(index=['gameId', 'playId', 'defendersInTheBox','numberOfPassRushers', 'DB', 'LB', 'DL',
                                           'yardline_first_dir', 'yardline_100_dir'], columns='posId',values='value')
        
        # Export the output to a CSV file based on the specified file path
        output_df.to_csv(fp)
        print(f"Defensive cleaning complete --- check assets/{fp}")
        return output_df