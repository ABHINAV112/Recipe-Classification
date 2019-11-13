# Intro to Data Science and Artificial Intelligence Project

A telegram bot to give a list of ingredients of food, given a picture and predict which cuisine the food belongs to. Also a neural network model to predict the nutrition values of food when given a name.

## Prerequisites

1. Python and jupyter notebook needs to be installed inorder to run the required code.
2. Run the following commands, to download the prerequisites

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install python-telegram-bot
pip install tensorflow
```

## Downloading the required data

1. Download the what's cooking dataset from [kaggle](https://www.kaggle.com/c/whats-cooking)
2. Download the glove embeddings text file from [kaggle](https://www.kaggle.com/terenceliu4444/glove6b100dtxt#glove.6B.100d.txt).
3. Make a folder called data, then proceed to move the train.json, the glove file and Food-Nutrition.xlsx into the data folder.

## Nutrient Vector Model

To run the nutrient vector prediction neural network, open the Nutrition_Training notebook. Now you may proceed to run each code segment as it goes down.

## Cuisine Prediction Model

To run the cusine prediction neural network, open the Cuisine_Training notebook. Now you may proceed to run each code segment and watch the output.

## Running the telegram bot

1. Make a folder called 'secret_token'.
2. Now, make a telegram bot using bot father. Add a file called bot_token.txt inside secret_token and paste the token into this file.
3. Using google cloud platform enable google cloud vision in a project and download the api key for the project in order to use google cloud vision. Place the key inside the secret key into secret_token and call it 'google_api_key.json'.
4. Now run the following line from the parent directory

```bash
python Telegram_Bot.py
```
