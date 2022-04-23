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

`get_positional_data()` combines all of the week-by-week positional data into one dataframe.

