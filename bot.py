from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents/intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['intent']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# message send

header = {
    "authorization": "USER DISCORD TOKEN"
}
import requests

channel = 'DISCORD CHANNEL ID'

# send a message based on latest message in discord channel
def send_discord_message(channelid):
    api = f"https://discord.com/api/v6/channels/{channelid}/messages"
    
    # get latest message from the channel
    r = requests.get(api, headers=header)
    #r = requests.get(api, headers=header)
    latest_message = ""
    if r.status_code == 200:
        latest_message = r.json()[11]['content'][0:]
        print(latest_message)
        res = chatbot_response(str(latest_message))
        payload = {"content": res}
        r = requests.post(api, data=payload, headers=header)
        if r.status_code == 200:
            print(f"message: {payload['content']} was sent") 
        
#send_discord_message(channel)

i = 0
while i <= 10:
    question = input()
    res = chatbot_response(question)
    print(res)
    i = i+1