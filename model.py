from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout   

def build_model(vocab_size, input_length, num_classes):
    inputs = Input(shape=(input_length,))

    x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)

    x = Bidirectional(LSTM(128))(x)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model