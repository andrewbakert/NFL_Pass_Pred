import pandas as pd
import numpy as np
def clean_positional(positions, first = 1, last = 17):
    # reading plays (see play data https://www.kaggle.com/c/nfl-big-data-bowl-2021/data)
    plays = pd.read_csv('nfl-big-data-bowl-2021/plays.csv')
    games = pd.read_csv('nfl-big-data-bowl-2021/games.csv')
        
    #to_datetime
    positions['time'] = pd.to_datetime(positions['time'], format='%Y-%m-%dT%H:%M:%S')
    #print(positions.columns)

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
    starting_pos_play_game['offdef'] = np.where((starting_pos_play_game['team'] == 'away') &
                                                (starting_pos_play_game['possessionTeam'] == starting_pos_play_game['visitorTeamAbbr']),
                                                'offense', 'defense')

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

    def find_rank(df, col, reverse=False):
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

    # Find the order of positions based on offensive direction.
    # First, group and extract first value of the y starting position and direction.
    pos_start = (starting_off_pers
                .groupby(['gameId', 'playId', 'position', 'nflId'])
                [['y_starting', 'x', 'off_pos', 'qb_side']].first()
                .reset_index())

    # Next, group and extract ranking of positions based on whether team is home or away
    # and the starting position.
    pos_order = np.where(pos_start['position'] != 'QB',
                        (pos_start.groupby(['gameId', 'playId', 'position', 'qb_side'])
                        .apply(lambda x: np.where(x.index.get_level_values(-1) == 'R',
                                                    find_rank(x, 'y_starting'),
                                                    find_rank(x, 'y_starting', reverse=True)))
                        .explode()
                        .values
                ),
                        (pos_start.groupby(['gameId', 'playId', 'position'])
                        .apply(lambda x: find_rank(x, 'y_starting'))
                        .explode()
                        .values
                        )
                        )

    # Add column with the position order to the df with indexed starting position.
    pos_start['pos_order'] = pos_order

    # Add number of position to position label to get position number.
    pos_start['pos_num'] = np.where(pos_start['position'] != 'QB',
                                    pos_start['position'].add(pos_start['qb_side']).add(pos_start['pos_order'].astype(str)),
                                    pos_start['position'].add(pos_start['pos_order'].astype(str)))

    #Adding a label of the players position (WR1, WR2). This makes sense from a numerical stand point, but shouldn't be used
    #to classify a team's WR1 WR2 etc.

    starting_off_pers = starting_off_pers.merge(pos_start[['gameId', 'playId', 'nflId', 'pos_num', 'pos_order']],
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
    cols = [col[:4] + '_in' for col in x_col]

    # Creating addition columns (boolean) for X player being in. If TE1 is in, flag says TRUE
    starting_pos[cols] = starting_pos[x_col].notnull()

    #Sparse Matrix
    starting_pos.fillna(0, inplace=True)

    #Final data! Everything is getting merged together.
    data = starting_pos.merge(starting_off_pers[['gameId', 'playId', 'offenseFormation']].drop_duplicates(),
                    left_index=True,
                    right_on=['gameId', 'playId'])
 
    data.dropna(axis=0, inplace=True)
    data = data.loc[:, ~np.all(data == 0, axis=0)]

    return data

