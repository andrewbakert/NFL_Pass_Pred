# NFL Big Data Bowl Pass Prediction

Following is a brief starting guide to replicating our work on predicting NFL passing quadrants based on offensive and defensive play and positional data. 

## Getting Started

### Clone the repo
First, to download the repo use the following command in Terminal: `git clone https://github.com/andrewbakert/NFL_Pass_Pred.git`

### Install required packages
Next, to download the requisite packages and versions run the following: `pip install -r requirements.txt`

### Download required data
We have provided a python script to download the required data from Kaggle and combine the week-by-week positional data into one dataframe. This file is located in [this script](get_data.py).

In order to download the data, run the following commands from that file:

`get_assets()` retrieves all data from Kaggle and stores the files in a folder named `Kaggle-Data-Files`.

`get_positional_data()` combines all of the week-by-week positional data into one dataframe. The positional data is given in x and y coordinates, and these coordinates are aligned with the field based on the diagram shown below.

![field positions](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3258%2F820e86013d48faacf33b7a32a15e814c%2FIncreasing%20Dir%20and%20O.png?generation=1572285857588233&alt=media)

## Visualizations
We also created several functions that can be used to visualize the positions of players on specific plays. One function creates a chart with starting positions for each player. A second function can be used to create an interactive chart with positions of each player at various time intervals that are averaged as quantile cuts. The process to create a starting diagram is shown below. The functions are in [play_vis.py](play_vis.py).
1) `positions = get_positional_data()`
2) `create_starting_chart(positions, gameId, playId)` The gameId and playId are integer representations of game and play IDs from the positional dataframe columns of the same names.

An example of an output of this function is shown below. Note that the x and y coordinate directions are reversed from the full field chart.
![starting vis](visualizations/first_play_starting.png)

## Data Cleaning
We separated data cleaning into cleaning for the offensive side and defensive side.

### Offensive Cleaning
The function used for offensive data cleaning is in [form_pred.py](form_pred.py). This function creates several features, specifically the starting positions of each offensive players, with null values where a position is not being used in a play. These nulls were filled with 0. The steps to process offensive data are shown below:
1) `positions = get_positional_data()`
2) `off_clean = clean_positional(positions)` There are several optional parameters for this function. `first` indicates the first week processed, `last` indicates the last week processed, and `yards_behind_line` indicates the number of yards used as the basis to flag whether players are far behind the line of scrimmage.

### Defensive Cleaning
The class used to process defensive data is available in [def_clean.py](def_clean.py). This class creates several features and returns additional features containing labels for each player by, with the options of blitz(`B`), zone(`Z`), or man(`M`) coverage. The code needed to generate these features are shown below:

`def_clean = DefensiveCleaning()` With default parameters, this will initialize the positional data and set attributes for all required data. Another attribute is `n_cuts`, which specifies the number of quantile cuts to group the positional data into in order to reduce dimensions.

`def_clean_df = def_clean.generate_full_df(1, 17)`. The first input is the first week processed, and the second is the last week processed.

## Pipeline Processing
Shown below is a visualization of the full pipeline used after processing the offensive and defensive data.
![pipe vis](visualizations/final_pipeline_vis.png)

### Ball Quadrants
In order to create a Y variable, we first processed data on where the ball was thrown. We separated the defensive side of the field into 16 quadrants, including 4 in the x direction and 4 in the y direction. The placement of the
