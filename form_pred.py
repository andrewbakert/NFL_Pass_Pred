import pandas as pd
import numpy as np
from datetime import datetime

def clean_positional(positions, first = 1, last = 17, yards_behind_line = 2):
    # reading plays (see play data https://www.kaggle.com/c/nfl-big-data-bowl-2021/data)
    plays = pd.read_csv('nfl-big-data-bowl-2021/plays.csv')
    games = pd.read_csv('nfl-big-data-bowl-2021/games.csv')

    #to_datetime
    positions['time'] = pd.to_datetime(positions['time'], format='%Y-%m-%dT%H:%M:%S')
    #print(positions.columns)

    #getting special teams plays
    events_to_remove_df = positions[positions['event'].isin(['qb_spike','punt_fake','field_goal_blocked','field_goal_fake','field_goal_play'])][['gameId','playId']].drop_duplicates()
    events_to_remove_df['ids'] = events_to_remove_df['gameId'].astype(str) + events_to_remove_df['playId'].astype(str)

    if (first != 1) or (last != 17):
        week_game_id = list(games[games['week'].isin(np.arange(first,last+1))]['gameId'].drop_duplicates())
        positions = positions[positions['gameId'].isin(week_game_id)]

    #get frame id of snap for each game and play id
    snap_frames = positions[positions['event'] == 'ball_snap'][['gameId','playId','frameId']]

    #get frame prior (unless snapped on frame 1)
    snap_frames['presnapId'] = snap_frames['frameId'].apply(lambda x: int(x)-1 if x>1 else x)

    #merge to remove all non frame snap -1 data
    presnap_df = positions.merge(snap_frames[['gameId','playId','presnapId']], left_on= ['gameId','playId','frameId'], right_on=['gameId','playId','presnapId'], how = 'right')

    # Get starting position of offensive players
    starting_pos = presnap_df.groupby(['gameId', 'playId', 'position', 'nflId', 'team'])[['x', 'y']].first().reset_index()

    # merging play data (see play data https://www.kaggle.com/c/nfl-big-data-bowl-2021/data)
    starting_pos_plays = starting_pos.merge(plays, on=['gameId', 'playId'], how='left')

    # data cleaning where yardline is not Null
    starting_pos_plays = starting_pos_plays[starting_pos_plays['absoluteYardlineNumber'].notnull()]
    # bring in game info (see game info data https://www.kaggle.com/c/nfl-big-data-bowl-2021/data)
    games = pd.read_csv('nfl-big-data-bowl-2021/games.csv')

    #bringing in features from games
    starting_pos_play_game = starting_pos_plays.merge(games, on='gameId', how='left')
    #naming which team has the ball as offense or defense
    starting_pos_play_game['offdef'] = np.where(
        ((starting_pos_play_game['team'] == 'away') &
         (starting_pos_play_game['possessionTeam'] == starting_pos_play_game['visitorTeamAbbr'])) |
        ((starting_pos_play_game['team'] == 'home') &
         (starting_pos_play_game['possessionTeam'] == starting_pos_play_game['homeTeamAbbr'])),
        'offense', 'defense')

    Dplayers_to_remove_df = starting_pos_play_game[(starting_pos_play_game['position'].isin(['CB','SS','FS','DL','DE','DT','DB','LB','MLB','OLB','ILB','NT','S'])) & (starting_pos_play_game['offdef'].isin(['offense']))][['gameId','playId']].drop_duplicates()
    Dplayers_to_remove_df['ids'] = starting_pos_play_game['gameId'].astype(str) + starting_pos_play_game['playId'].astype(str)

    #starting position from offense players
    starting_off = starting_pos_play_game[starting_pos_play_game['offdef'] == 'offense']

    # What personal is on the field
    personnel = starting_off['personnelO'].str.extract('(?P<RB>\d+)\sRB\,\s(?P<TE>\d+)\sTE\,\s(?P<WR>\d+)\sWR')
    personnel = personnel.astype(float)

    # Adding that as a feature in the new DF
    starting_off_pers = pd.concat([starting_off, personnel], axis=1)

    # Subtracting 10 because the endzone adds 10 years to field
    starting_off_pers['yardline_100'] = starting_off_pers['absoluteYardlineNumber'].sub(10)

    # If position X is less than yardline100, return yardline100 - starting position, else, starting position - yardline.
    # This gets # of yards behind line no matter which way they are facing.

    # Y starting is the y coords of the starting position.
    starting_off_pers['off_pos'] = np.where(starting_off_pers['x'].lt(starting_off_pers['absoluteYardlineNumber']), 'left', 'right')
    starting_off_pers['x_behind_line'] = np.where(starting_off_pers['off_pos'] == 'right',
                                                  starting_off_pers['absoluteYardlineNumber'].sub(starting_off_pers['x']),
                                                  starting_off_pers['x'].sub(starting_off_pers['absoluteYardlineNumber']))
    starting_off_pers['y_starting'] = np.where(starting_off_pers['off_pos'] == 'right',
                                               starting_off_pers['y'].rsub(53.3), starting_off_pers['y'])

    # Y QB is the y starting position of the quarterback.
    starting_off_pers['y_qb'] = starting_off_pers.groupby(['gameId', 'playId']).apply(lambda x: np.repeat(53.3/2, x.shape[0])
    if x[x['position'] == 'QB'].shape[0] == 0 else np.repeat(x[x['position'] == 'QB']['y_starting'].iloc[0], x.shape[0])).explode().values
    starting_off_pers['y_qb'] = starting_off_pers['y_qb'].astype(float)

    # Find side of player relative to QB and the starting y coordinates relative to the QB.
    starting_off_pers['qb_side'] = np.where(starting_off_pers['y_starting'].gt(starting_off_pers['y_qb']), 'R', 'L')
    starting_off_pers['y_starting_qb'] = starting_off_pers['y_starting'].sub(starting_off_pers['y_qb'])

    # Find the order of positions based on offensive direction.
    # First, group and extract first value of the y starting position and direction.
    pos_start = (starting_off_pers
                 .groupby(['gameId', 'playId', 'position', 'nflId'])
                 [['y_starting', 'x', 'off_pos', 'qb_side']].first()
                 .reset_index())

    # Next, group and extract ranking of positions based on whether team is home or away
    # and the starting position.
    qb_start = pos_start[pos_start['position'] == 'QB']
    non_qb_start = pos_start[pos_start['position'] != 'QB']
    left = non_qb_start[non_qb_start['qb_side'] == 'L']
    right = non_qb_start[non_qb_start['qb_side'] == 'R']
    l = left.sort_values('y_starting',ascending=False).groupby(['gameId', 'playId', 'position']).apply(
        lambda x: list(zip(x['nflId'], range(x.shape[1])))).explode().reset_index().rename({0: 'nfl_num'}, axis=1)
    r = right.sort_values('y_starting').groupby(['gameId', 'playId', 'position']).apply(
        lambda x: list(zip(x['nflId'], range(x.shape[1])))).explode().reset_index().rename({0: 'nfl_num'}, axis=1)
    qb = qb_start.sort_values('y_starting').groupby(['gameId', 'playId', 'position']).apply(
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
    qb_full = qb.merge(pos_start, on=['gameId', 'playId', 'nflId', 'position'])
    full_w_qb = pd.concat([full, qb_full[['gameId', 'playId', 'position',
                                          'qb_side', 'nflId', 'pos_order']]], axis=0)
    start_df_full = pos_start.merge(full_w_qb, on=['gameId', 'playId', 'nflId', 'position', 'qb_side'])

    start_df_full['pos_num'] = np.where(start_df_full['position'] != 'QB',
                                        start_df_full['position'].add(start_df_full['qb_side']).add(start_df_full['pos_order'].astype(str)),
                                        start_df_full['position'].add(start_df_full['pos_order'].astype(str)))

    #Adding a label of the players position (WR1, WR2). This makes sense from a numerical stand point, but shouldn't be used
    #to classify a team's WR1 WR2 etc.

    starting_off_pers = starting_off_pers.merge(start_df_full[['gameId', 'playId', 'nflId', 'pos_num', 'pos_order']],
                                                on=['gameId', 'playId', 'nflId'])

    # Convert to matrix of GameID and PlayID. Grab number of yards behind line for each player.
    starting_x = (starting_off_pers
                  .pivot_table(columns='pos_num', index=['gameId', 'playId'], values='x_behind_line').rename(lambda x: x + '_x', axis=1))

    #Same as above, but for Y coords relative to the QB.
    starting_y = (starting_off_pers
                  .pivot_table(columns='pos_num', index=['gameId', 'playId'], values='y_starting_qb').rename(lambda x: x + '_y', axis=1))

    #merging to get coords of players with _X and _Y
    starting_pos = starting_x.merge(starting_y, left_index=True, right_index=True)

    #X_col is getting all the X columns. Cols is creating a list that say "WR1_in", "FB1_in" etc
    x_col = starting_pos.columns[starting_pos.columns.str.match('.*\_x$')]
    cols = [col[:4] + '_in' if col[:2] != 'QB' else col[:3] + '_in'for col in x_col]

    # Creating addition columns (boolean) for X player being in. If TE1 is in, flag says TRUE
    starting_pos[cols] = starting_pos[x_col].notnull()

    #Sparse Matrix
    starting_pos.fillna(0, inplace=True)

    #Final data! Everything is getting merged together.
    data = starting_pos.merge(starting_off_pers[['gameId', 'playId', 'offenseFormation']].drop_duplicates(),
                              left_index=True,
                              right_on=['gameId', 'playId'])

    ids_to_remove_list = events_to_remove_df['ids'].to_list() + Dplayers_to_remove_df['ids'].to_list() 

    data['gamePlayId'] = data['gameId'].astype(str) + data['playId'].astype(str)
    data = data[~data['gamePlayId'].isin(ids_to_remove_list)]

    data.dropna(axis=0, inplace=True)
    data = data.loc[:, ~np.all(data == 0, axis=0)]
    data['on_left'] = data.iloc[:, data.columns.str.contains('L\d_in')].sum(axis=1)
    data['on_right'] = data.iloc[:, data.columns.str.contains('R\d_in')].sum(axis=1)
    data['perc_left'] = data['on_left'].div(data['on_left'].add(data['on_right']))
    data['perc_right'] = data['on_right'].div(data['on_left'].add(data['on_right']))
    data.drop(['on_left', 'on_right'], axis=1, inplace=True)
    data['perc_behind_los'] = data.loc[:, data.columns.str.contains('x')].apply(
        lambda x: x[x<=-yards_behind_line].shape[0] / x[x != 0].shape[0], axis=1)
    data_in = data.loc[:, data.columns.str.contains('\d_in')]
    data_in_stacked = data_in.stack().reset_index()
    data_in_stacked.columns = ['idx', 'position', 'in']
    data_in_stacked['position'] = data_in_stacked['position'].str.extract('^([A-Z]{2})')
    data_in_pos = data_in_stacked.groupby(['idx', 'position'])['in'].sum().reset_index()
    data_pos_pivot = data_in_pos.pivot_table(values='in', columns='position', index='idx')
    data_full = data.merge(data_pos_pivot, left_index=True, right_index=True)

    addit_feat = starting_pos_play_game[starting_pos_play_game['offdef'] == 'offense'][['team','gameId','playId']].drop_duplicates().merge(plays[['gameId','playId','preSnapVisitorScore','preSnapHomeScore','gameClock', 'possessionTeam', 'down','quarter']], left_on = ['gameId','playId'], right_on = ['gameId','playId'])
    addit_feat['score_differential'] = np.where(addit_feat['team'] == 'away', addit_feat['preSnapVisitorScore'] - addit_feat['preSnapHomeScore'],   addit_feat['preSnapHomeScore'] - addit_feat['preSnapVisitorScore'])
    addit_feat['score_differential'] = np.where(addit_feat['team'] == 'away', addit_feat['preSnapVisitorScore'] - addit_feat['preSnapHomeScore'],   addit_feat['preSnapHomeScore'] - addit_feat['preSnapVisitorScore'])
    def get_sec(time_str):
        """Get seconds from time."""
        try:
            m, s, _ = time_str.split(':')
            return int(m) * 60 + int(s)
        except:
            return 0
    addit_feat['timeRemaining'] = (4 - addit_feat['quarter']) * 900 + addit_feat['gameClock'].apply(lambda x: get_sec(str(x)))
    data_full = data_full.merge(addit_feat[['score_differential','possessionTeam', 'down','timeRemaining','gameId','playId']], how = 'left', left_on = ['gameId','playId'], right_on = ['gameId','playId'])
    return data_full

