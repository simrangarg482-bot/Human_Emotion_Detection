import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import preprocess
from model import build_model
from decision_enginee import get_recommendation

# ------------------ CONFIG ------------------
MAX_LEN = 50
VOCAB_SIZE = 10000
EPOCHS = 50
BATCH_SIZE = 32

# ------------------ LOAD DATA ------------------
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# ------------------ PREPROCESS ------------------
print("Preprocessing...")
train_df, encoders, target_encoder = preprocess(train_df)

# ------------------ TOKENIZATION ------------------
print("Tokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['journal_text'])

# Convert text → sequences
train_sequences = tokenizer.texts_to_sequences(train_df['journal_text'])

# Padding
X_train_pad = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post')

# Labels
y_train = train_df['emotional_state'].values
num_classes = len(np.unique(y_train))

# ------------------ BUILD MODEL ------------------
print("Building model...")
model = build_model(
    vocab_size=VOCAB_SIZE,
    input_length=MAX_LEN,
    num_classes=num_classes
)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, class_weights))

# ------------------ TRAIN ------------------
print("Training model...")
model.fit(
    X_train_pad,
    y_train,
    epochs=50,   # increase
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weights   # VERY IMPORTANT
)

# ------------------ TEST PREPROCESS ------------------
print("Processing test data...")
test_df, _, _ = preprocess(
    test_df,
    encoders=encoders,
    target_encoder=target_encoder,
    is_train=False
)

test_sequences = tokenizer.texts_to_sequences(test_df['journal_text'])
X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post')

# ------------------ PREDICTION ------------------
print("Predicting...")
preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)

# Convert back to emotions
pred_emotions = target_encoder.inverse_transform(pred_labels)

# ------------------ SAVE BASIC PREDICTIONS ------------------
submission_df = test_df.copy()
submission_df['predicted_emotion'] = pred_emotions

submission_df.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")

# ------------------ DECISION ENGINE ------------------
print("Generating recommendations...")
results = []

for emotion in pred_emotions:
    intensity = 3  # placeholder (you can upgrade later)

    emotion_out, intensity_out, action, timing = get_recommendation(
        emotion, intensity
    )

    results.append((emotion_out, intensity_out, action, timing))

# Add results to dataframe
submission_df['predicted_emotion'] = [r[0] for r in results]
submission_df['intensity'] = [r[1] for r in results]
submission_df['recommended_action'] = [r[2] for r in results]
submission_df['recommended_time'] = [r[3] for r in results]

# ------------------ SAVE FINAL OUTPUT ------------------
submission_df.to_csv("final_output.csv", index=False)
print("Saved final_output.csv")

# ------------------ SAVE RECOMMENDATIONS ONLY ------------------
recommendations_df = submission_df[
    ['predicted_emotion', 'intensity', 'recommended_action', 'recommended_time']
]

recommendations_df.to_csv("recommendations.csv", index=False)
print("Saved recommendations.csv")

# ------------------ SAVE TOKENIZER ------------------
tokenizer_json = tokenizer.to_json()

with open("tokenizer.json", "w") as f:
    f.write(tokenizer_json)

# ------------------ SAVE MODEL ------------------
model.save("model.h5")   # ✅ FULL MODEL (architecture + weights)
print("Saved model.h5")

# ------------------ SAVE LABEL ENCODER ------------------
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

print("Saved label_encoder.pkl")

# ------------------ DONE ------------------
print("✅ Training + Prediction Pipeline Completed Successfully!")
print("Number of classes:", num_classes)
print(train_df['emotional_state'].value_counts())  

def build_model(vocab_size, input_length, num_classes):
    inputs = Input(shape=(input_length,))

    x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)

    x = Bidirectional(LSTM(128))(x)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model