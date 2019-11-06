# Cuisine prediction model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from wget import download

raw_data_frame = pd.read_json('data/train.json')
raw_data_frame.head()

def text_prepare(ingredient):
    import re
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    ingredient = re.sub(REPLACE_BY_SPACE_RE,' ',ingredient)
    ingredient = ingredient.lower()
    ingredient = ingredient.strip()
    ingredient = ' '.join([word for word in ingredient.split(" ")])
    return ingredient

def loader(ingredient_list):    
    return ' '.join([text_prepare(ingredient) for ingredient in ingredient_list])


raw_data_frame["ingredients"] = raw_data_frame['ingredients'].apply(loader)
raw_data_frame["ingredients"][:5]

cuisines = raw_data_frame['cuisine'].unique()
cuisine_labels = dict()
for i,val in enumerate(cuisines):
    cuisine_labels[i] = val
cuisine_labels


ingredients = np.array(raw_data_frame['ingredients'])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(ingredients)
vocabulary_size = len(tokenizer.word_index) + 1

word_index = tokenizer.word_index

def word_input_form(word):
    token = tokenizer.texts_to_sequences([word])
    padded = pad_sequences(token,maxlen=40,padding='post',dtype='int32')
    return padded


model = models.load_model('model/cuisine_prediction.h5')

def predict(word):
    input_word = word_input_form(word)
    out = np.argmax(model.predict(input_word))
    return(cuisine_labels[out])

########################################################
# google cloud vision
def detect_labels(path):
    import io
    """Detects labels in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    # print(labels)
    tag_set=  set()
    tag_list = list()
    for label in labels:
        curr_description = set(label.description.split(' '))
        tag_set = tag_set.union(curr_description)
        tag_list.append(label.description)

    ingredients = []
    for i,val in enumerate(tag_list):
        val = val.lower()
        if( val in word_index):
            ingredients.append(val)


    tag_string = ' '.join(list(tag_set))
    return tag_string , ingredients
    


########################################################
# Telegram bot 
with open('secret_token/bot_token.txt') as t:
    token = t.read().rsplit()[0]
print(token)
updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher

def start(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Welcome to Cuisine Bot!")

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

# function handler for help command
def help(update,context): 
    context.bot.send_message(chat_id=update.message.chat_id, text="1. Send us an image as a 'photo' in the telegram bot, and receive the ingredients and cuisines of the dish\n2. Send us a sentence of space separated integers and you can find out a cuisine which you can make with the given ingredients!")

help_handler = CommandHandler('help',help)
dispatcher.add_handler(help_handler)

def handle_images(update,context):
    file = context.bot.getFile(update.message.photo[-1].file_id)
    file_path = file.file_path
    print("got here")
    # location to download the image
    image_path = "./Telegram_Images"

    # downloading the image using wget, assigning the file path to downloaded_file
    downloaded_file = download(file_path,image_path)
    labs, ingredients = detect_labels("test_images/Creamy-Tomato-and-Spinach-Pasta-skillet-1-500x480.jpg")
    out_ingredients = 'Ingredients:\n'
    for i,val in enumerate(ingredients):
        out_ingredients += str(i+1) +'. '+val+'\n'
    cuisine = "Cuisine:\n"+predict(labs)
    print(out_ingredients,cuisine,sep='\n')
    context.bot.send_message(chat_id=update.message.chat_id,text = out_ingredients)
    context.bot.send_message(chat_id=update.message.chat_id,text = cuisine)

image_handler = MessageHandler(Filters.photo,handle_images)
dispatcher.add_handler(image_handler)

def handle_text(update,context):
    curr_msg = update.message.text
    output_prediction = predict(curr_msg)
    print(output_prediction)
    context.bot.send_message(chat_id=update.message.chat_id,text = output_prediction)

text_handler = MessageHandler(Filters.text,handle_text)
dispatcher.add_handler(text_handler)

updater.start_polling()

