import spacy
import string
import pandas as pd
from rapidfuzz import process, fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pprint

nlp = spacy.load('en_core_web_sm') #for extracting keywords

model_name = "bhadresh-savani/distilbert-base-uncased-emotion" #for emotions

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
 
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

df = pd.read_csv('indian_food.csv')
df = df.fillna('')

#keywords
filter_keywords = {
    "diet": ['vegetarian', 'non-vegetarian'],
    "state": ['West Bengal','Rajasthan','Punjab','Uttar Pradesh','Odisha',
 'Maharashtra','Uttarakhand','Assam','Bihar','Andhra Pradesh','Karnataka',
 'Telangana','Kerala','Tamil Nadu','Gujarat','Tripura','Manipur',
 'Nagaland','Delhi','Jammu', 'Kashmir','Chhattisgarh','Haryana',
 'Madhya Pradesh','Goa'],
    "flavor_profile": ['sweet' ,'spicy' ,'bitter','sour'],
    "region": ['East' ,'West','North' ,'North East' ,'South' ,'Central'],
    "course": ['dessert' ,'main course' ,'starter' ,'snack']
    
}

mood_map = {
    "sadness": {"flavor_profile": ["sweet"], "course": ["dessert", "main course"]},
    "joy": {"flavor_profile": ["spicy"], "course": ["snack", "starter"]},
    "anger": {"flavor_profile": ["cooling", "mild"], "diet": ["vegetarian"]},
    "love": {"flavor_profile": ["sweet"], "course": ["dessert"]},
    "surprise": {"flavor_profile": ["spicy"], "course": ["snack"]},
    "neutral": {"course": ["main course"]}
}
 
def get_mood(prompt):
    emotions = emotion_classifier(prompt)[0]
    top_emotion = sorted(emotions, key=lambda x: x['score'], reverse=True)[0]['label']
    return top_emotion  # e.g., 'joy', 'sadness', 'anger', etc.

def spacy_tokenizer(sentence):
    stop_words = nlp.Defaults.stop_words #for removing stop words
    punctuations = string.punctuation #for removing punctuations
    doc = nlp(sentence)
    myTokens = [word.lemma_.lower().strip() for word in doc] #lemmatization
    myTokens = [word for word in myTokens if word not in stop_words and word not in punctuations]
    #print(myTokens)
    return myTokens

def map_keywords_to_filters(user_keywords): 
    filters={}
    for keyword in user_keywords:
        keyword = keyword.lower()
        
        for field, values in filter_keywords.items():
            # #print(process.extractOne(keyword, values))
            match, score, _ = process.extractOne(keyword, values, scorer=fuzz.token_set_ratio)
            if(score>75): #extracts words matching keyword
                filters[field] = match
                break
    return filters

def filtered_output(filters): 
    #print(filters)
    df_filtered = df.copy()
    for col, value in filters.items():
        value = value.strip().lower()
        #print(col, value)
        if col in df_filtered:
            df_filtered = df_filtered[df_filtered[col].str.lower() == value] #filters the dataset
    #print('df', df_filtered)
    return pprint.pformat(df_filtered[['name', 'ingredients', 'diet', 'prep_time', 'cook_time','flavor_profile','course', 'state', 'region']].to_dict(orient="records"))

def get_dishes_by_region(user_input):
    # Process user input with spaCy
    return filtered_output(map_keywords_to_filters(spacy_tokenizer(user_input)))

def get_dishes_by_mood(user_input):
    mood = get_mood(user_input)
    #print(f"Detected mood: {mood}")

    filters = mood_map.get(mood, {})
    
    filtered_df = df.copy()
    
    for key, values in filters.items():
        filtered_df = filtered_df[filtered_df[key].str.lower().isin([v.lower() for v in values])]
        #print(filtered_df)
    return filtered_df[['name', 'ingredients', 'diet', 'prep_time', 'cook_time','flavor_profile','course', 'state', 'region']].to_dict(orient="records")