def get_assets():
    ## instructions on getting data through Kaggle API. API key is read from user/'your name'/.kaggle directory on a mac.
    ## https://www.kaggle.com/general/74235
    import os
    import subprocess
    import sys
    print("")
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
    cwd = str(os.getcwd())
    with zipfile.ZipFile(cwd + '/nfl-big-data-bowl-2021.zip', 'r') as zip_ref:
        zip_ref.extractall(cwd + '/Kaggle-Data-Files')

    print("Data Successfully Downloaded")

def get_positional_data():
    import pandas as pd
    import numpy as np
    import os

    dir = 'assets'
    fp = dir + '/full_position.csv'
    cwd = str(os.getcwd())
    if not os.path.exists(dir):
        os.mkdir(dir)
        positions = pd.DataFrame()
        for week in range(1, 18):
            week = pd.read_csv(cwd + f'/Kaggle-Data-Files/week{week}.csv')
            positions = pd.concat([positions, week], axis=0)
        positions.to_csv(fp, index=False)
    else:
        positions = pd.read_csv(fp)
    return positions