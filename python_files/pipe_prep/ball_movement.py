def ball_quadrants(pos, y_sections = 3):
    '''
    Return dataframe pos with the quadrant of where the ball is thrown.
    y_sections: the number of y buckets for quadrant creation

    Parameters
    ----------
    pos : Pandas DataFrame
        Dataframe of all positions and plays created from clean_positional()
    y_sections : int
        the number of y buckets for quadrant creation
   
    Returns
    -------
    The pos dataframe with additional parameters including the x and y vector space of a given pass and its associated quadrant.
    '''
    #imports
    import pandas as pd
    import numpy as np
    plays = pd.read_csv('../../nfl-big-data-bowl-2021/plays.csv')

    #Get position of ball when it is caught, incomplete, intercepted, etc.
    pass_end = pos[pos['event'].isin(['pass_arrived','pass_outcome_caught','pass_outcome_incomplete','pass_outcome_interception','pass_outcome_touchdown'])].query("displayName == 'Football'").groupby(['gameId','playId']).first().reset_index()[['gameId','playId','x','y','playDirection','event']]
    #get position of ball when it is snapped
    pass_start = pos[pos['event'].isin(['ball_snap'])].query("displayName == 'Football'").groupby(['gameId','playId']).first().reset_index()[['gameId','playId','x','y','playDirection','event']]

    #the below statements are grabbing the x and y as if the offense was always moving from left to right
    pass_start['x_zero_base'] = np.where(pass_start['playDirection'] == 'left', 120-pass_start['x'], pass_start['x'])
    pass_start['y_zero_base'] = np.where(pass_start['playDirection'] == 'left', 53.3-pass_start['y'], pass_start['y'])
    pass_end['x_zero_base'] = np.where(pass_end['playDirection'] == 'left', 120-pass_end['x'], pass_end['x'])
    pass_end['y_zero_base'] = np.where(pass_end['playDirection'] == 'left', 53.3-pass_end['y'], pass_end['y'])

    #merging the starting and ending datafraems together
    pass_df = pass_start.merge(pass_end, left_on = ['gameId','playId'],  right_on = ['gameId','playId'], suffixes = ['0','1'])[['gameId','playId','x0','y0','x_zero_base0','y_zero_base0','x1','y1','x_zero_base1','y_zero_base1']]
        
    #creating the x and y values for the vector of thrown pass
    pass_df['x_vec'] = pass_df['x_zero_base1'] - pass_df['x_zero_base0'] 
    pass_df['y_vec'] = pass_df['y_zero_base1'] - pass_df['y_zero_base0'] 

    #adding additional features including play description (For interactive viz), down, distance, etc.
    pass_df = pass_df.merge(plays[['gameId','playId','passResult','playDescription', 'down','yardsToGo']], left_on = ['gameId','playId'], right_on=['gameId','playId'])
    pass_df['passResult'] = np.where(pass_df['playDescription'].str.contains('TOUCHDOWN'),"TD",pass_df['passResult'])

    #this function bins x and y coords into the appropriate quadrant
    def create_quadrants(df, y_sections):
        #make copy of df 
        ret_df = df 
        #create array of all y field space
        y_sec_array = np.array([-53.5] + list(np.arange(-53.5/2,53.5/2+1,53.5/y_sections)[1:-1]) + [53.5])

        #I created these arbitrary breakouts for X quadrant.
        #behind LOS, LOS to line to gain (ltg), ltg + 10, beyond ltg + 10. 
        #This is completely arbitrary and can be adjusted.  
        ret_df['x_sec_array'] = ret_df['yardsToGo'].apply(lambda x: np.array([-100,0,x,x+10,x+110]))
                
        def get_xquad(xpass,arr):
            q = 0
            for i in np.array(arr):
                if xpass <= i:
                    val = q
                    break
                else:
                    q += 1
            return(val)

        ret_df['x_quad'] = ret_df.apply(lambda x: get_xquad(x['x_vec'], x['x_sec_array']), axis = 1)

        #bucket each y coord into bin
        ret_df['y_bucket'] = pd.cut(ret_df['y_vec'], y_sec_array)

        #create a lookup dict for quad num
        ybucket_dict = {v: k for k, v in enumerate(ret_df['y_bucket'].drop_duplicates().sort_values())}

        #apply quad num as column y_quad
        ret_df['y_quad'] = ret_df['y_bucket'].apply(lambda x: ybucket_dict.get(x))

        return ret_df

    #running func
    quad_df = create_quadrants(pass_df, y_sections=y_sections)

    #converting from obj to int
    quad_df['x_quad'] = quad_df['x_quad'].astype(int)
    quad_df['y_quad'] = quad_df['y_quad'].astype(int)

    #dropping arrays because the vizualizations won't run with them
    return quad_df.drop(['x_sec_array','y_bucket'], axis = 1)
        
def make_quad_chart(pass_df):
    '''Create a simple visualization to see a heatmap of quadrants
    
    Parameters
    ----------
    pass_df : Pandas DataFrame
        dataframe created from ball_quadrant function
    
    Returns
    ----------
    Altair heatmap of quadrant frequency from pass_df '''

    
    #import altair for visualization
    import altair as alt
    alt.data_transformers.disable_max_rows()
    #Chreate and return the simple chart
    return alt.Chart(pass_df[['x_quad','y_quad']]).mark_rect().encode(
        x='x_quad:O',
        y='y_quad:O',
        color = 'count():Q'
    ).properties(
        width=400,
        height=300,
        title={
        "text": ["Ball thrown Quadrant"], 
        "subtitle": ["X_quad1 = behind LOS",
        "x_quad2 = between LOS and line to gain",
        "x_quad3 = between line to gain and additional 10 yards",
        "x_quad4 = farther than line to gain + 10 yards"],
        }
    )