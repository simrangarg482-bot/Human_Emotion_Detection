import pandas as pd
import numpy as np 
import re
from sklearn.preprocessing import LabelEncoder 
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv") 
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower().strip() #strip removes beginning and ending spaces 
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s!?]', '', text)
    
    return text 
def process_text(df):
    df['journal_text'] = df['journal_text'].apply(clean_text)
    return df 
def handle_missing_values(df):
    
    numeric_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
    categorical_cols = ['ambience_type', 'time_of_day', 'previous_day_mood', 
                        'face_emotion_hint', 'reflection_quality']
    
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    df[col] = df[col].fillna(df[col].median()) #we have filled median instead of mean so that missing values may not get affected by outliers
    
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "unknown"
    df[col] = df[col].fillna("unknown")

    return df 
def encode_features(df, encoders=None):

    categorical_cols = ['ambience_type', 'time_of_day', 'previous_day_mood', 
                        'face_emotion_hint', 'reflection_quality']

    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()

            df[col] = df[col].astype(str)
            df[col] = df[col].fillna("unknown")

            # Add unknown class
            unique_values = list(df[col].unique())
            if "unknown" not in unique_values:
                unique_values.append("unknown")

            le.fit(unique_values)

            df[col] = le.transform(df[col])
            encoders[col] = le

    else:
        for col in categorical_cols:
            df[col] = df[col].apply(
                lambda x: x if x in encoders[col].classes_ else "unknown"
            )
            df[col] = encoders[col].transform(df[col])

    return df, encoders   # 🔥 THIS LINE WAS MISSING
def encode_target(df, target_encoder=None):
    
    if target_encoder is None:
        target_encoder = LabelEncoder()
        df['emotional_state'] = target_encoder.fit_transform(df['emotional_state'])
    else:
        df['emotional_state'] = target_encoder.transform(df['emotional_state'])
    
    return df, target_encoder 
def preprocess(df, encoders=None, target_encoder=None, is_train=True):
    
    df = process_text(df)
    df = handle_missing_values(df)
    df, encoders = encode_features(df, encoders)
    
    if is_train:
        df, target_encoder = encode_target(df, target_encoder)
    
    return df, encoders, target_encoder
result = preprocess(train_df)
print(result)