import sys
sys.path.append('../../NFL_Pass_Pred')

def get_assets():
    '''Get NFL data from Kaggle API and store in working directory'''
    ## instructions on getting data through Kaggle API. API key is read from user/'your name'/.kaggle directory on a mac.
    ## https://www.kaggle.com/general/74235
    import os
    import subprocess
    
    if os.path.exists('../Kaggle-Data-Files/week17.csv'):
        print('assets previously downloaded')
    else: 
        print("This process will pip install Kaggle and download data through Kaggle API.")
        print("")
        print("Please confirm that you've downloaded Kaggle JSON credentials into directory")
        print("")
        confirm = input("Type 'Y' to continue: ")
        if confirm.lower() != "y":
            print("not confirmed. Aborting.")
            quit()
        subprocess.run(["pip", "install", "kaggle"])
        subprocess.run(["kaggle", "competitions", "download", "-c", 'nfl-big-data-bowl-2021'])

        import zipfile

        with zipfile.ZipFile('../nfl-big-data-bowl-2021.zip', 'r') as zip_ref:
            zip_ref.extractall('../Kaggle-Data-Files')

        print("Data Successfully Downloaded")

def get_positional_data():
    '''Consolidate KAggle API data into one working file called positions'''
    import pandas as pd
    import numpy as np
    import os
    print(os.getcwd())
    
    dir = '../assets'
    fp = dir + '/full_position.csv'
    cwd = str(os.getcwd())
    if not os.path.exists(dir):
        print('creating directory')
        os.mkdir(dir)
    if os.path.exists(fp):
        print('positional data already downloaded.')
        print('reading positional data.')
        positions = pd.read_csv(fp)
        print('returning positional data.')
    else:
        print('reading data')
        positions = pd.DataFrame()
        for week in range(1, 18):
            print(f'reading week {week}')
            week_data = pd.read_csv(f'../Kaggle-Data-Files/week{week}.csv')
            week_data['week'] = week
            positions = pd.concat([positions, week_data], axis=0)
            print(f'added week {week}')
        positions.to_csv(fp, index=False)
        print(f'positional data written to {fp}')
        print('returning positional data.')
    return positions