import pandas as pd
import numpy as np
from get_data import get_positional_data
import altair as alt

def create_starting_chart(pos_df, game_id, play_id):
    """
    Creates chart containing position of each player for a given game and play.
    
    Parameters
    ----------
    pos_df : Pandas DataFrame
        Dataframe of all positions and plays
    game_id : str
        ID of game of extracted play
    play_id : str
        ID of play of extracted play
        
    Returns
    -------
    Altair Chart with player positions, the line of scrimmage, and the first yardline.
    """

    # Extract all plays.
    plays_df = pd.read_csv('Kaggle-Data-Files/plays.csv')

    # Narrow positions to include only given game and play.
    play_df = pos_df[(pos_df['gameId'] == game_id) & (pos_df['playId'] == play_id)]

    # Extract data only after ball is snapped and merge with plays.
    after_snap = play_df[play_df['time'].ge(play_df[play_df['event'] == 'ball_snap']['time'].iloc[0])]
    after_snap_play = after_snap.merge(plays_df, on=['gameId', 'playId'])

    # Find yardline with 100 yards as the basis.
    after_snap_play['x_100'] = after_snap_play['x'].sub(10)

    # Find offensive and defensive mapping based on what team QB is on.
    off_def_map = after_snap_play.groupby('team').apply(
        lambda x: 'QB' in x['position'].values).reset_index().rename({0: 'off'}, axis=1)
    off_def_map.replace({True: 'Offense', False: 'Defense'}, inplace=True)

    # Merge offensive and defensive mapping with main dataframe and calculate
    # 100-based line of scrimmage.
    after_snap_off = after_snap_play.merge(off_def_map, on='team')
    after_snap_off['yardline_100'] = after_snap_off['absoluteYardlineNumber'].sub(10)

    # Find which side the offense is lined up on.
    try:
        on_left = after_snap_off[after_snap_off['position'] == 'QB'][
                      'x_100'].iloc[0] < after_snap_off['yardline_100'].iloc[0]
    except IndexError:
        try:
            on_left = (after_snap_off[after_snap_off['position'] == 'P']['x_100'].iloc[0]
                       < after_snap_off['yardline_100'].iloc[0])
            team_map = after_snap_off.groupby(['gameId', 'playId', 'team']).apply(lambda x: 'P' in x['position'].unique())
            team_map = team_map.map(lambda x: 'Offense' if x else 'Defense')
            team_map.name = 'off'
            after_snap_off.drop('off', axis=1, inplace=True)
            after_snap_off = after_snap_off.merge(
                team_map, left_on=['gameId', 'playId', 'team'], right_index=True)
        except IndexError:
            on_left = after_snap_off['x_100'].iloc[0] > after_snap_off['yardline_100'].iloc[0]


    if on_left == False:
        side = 'right'
    else:
        side = 'left'

        # If lined up on left, add yards to go to find first down marker. Otherwise, subtract.
    if side == 'left':
        after_snap_off['yardline_first'] = after_snap_off['yardline_100'].add(after_snap_off['yardsToGo'])
    else:
        after_snap_off['yardline_first'] = after_snap_off['yardline_100'].sub(after_snap_off['yardsToGo'])


    # Find starting positions.
    starting = after_snap_off.groupby(['nflId'])[
        ['x_100', 'y', 'position', 'yardline_100', 'yardline_first', 'off']].first().reset_index()

    # Set directional sides based on where offense is lined up.
    if side == 'right':
        starting['x_dir'] = starting['x_100'].rsub(100)
        starting['yardline_dir'] = starting['yardline_100'].rsub(100)
        starting['yardline_first_dir'] = starting['yardline_first'].rsub(100)
        starting['y_dir'] = starting['y'].copy()
    else:
        starting['x_dir'] = starting['x_100'].copy()
        starting['yardline_dir'] = starting['yardline_100'].copy()
        starting['yardline_first_dir'] = starting['yardline_first'].copy()
        starting['y_dir'] = starting['y'].rsub(53.3)
    alt.themes.enable('fivethirtyeight')

    # Create chart for marks of positions.
    marks = (alt.Chart(starting)
             .encode(x=alt.X('y_dir:Q', scale=alt.Scale(domain=[0, 53.3])),
                     y=alt.Y('x_dir:Q'),
                     color=alt.Color('off:N', title=None))
             .mark_circle(size=100, dx=5, dy=5)
             )

    # Create chart for position labels.
    labels = (alt.Chart(starting)
              .encode(x=alt.X('y_dir:Q', scale=alt.Scale(domain=[0, 53.3]), title=None),
                      y=alt.Y('x_dir:Q', scale=alt.Scale(zero=False), title='Yardline'),
                      text=alt.Text('position:N'))
              .mark_text(dx=-15)
              )

    # Create charts for first down and line of scrimmage.
    yardline_100 = alt.Chart(starting).encode(y=alt.Y('yardline_dir:Q')).mark_rule(color='blue')
    yardline_first = alt.Chart(starting).encode(y=alt.Y('yardline_first_dir:Q')).mark_rule(color='#ECEC01')
    full_chart = ((marks + labels + yardline_100 + yardline_first)
                  .properties(width=53.3*15, height=(starting['x_dir'].max() - starting['x_dir'].min())*15))
    return full_chart