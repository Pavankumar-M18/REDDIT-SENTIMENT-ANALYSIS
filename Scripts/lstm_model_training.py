# Importing Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# --------------------  Data preparation  ------------------------


# Load the dataset
df = pd.read_csv('Data/reddit_sentiment_results.csv')

x = df['Cleaned_Text'].astype(str)           # Ensure all text is string
y = df['VADER_Label'].str.lower()            # Normalize labels (e.g., Positive → positive)


# Encode labels (Convert positive, Negative, Neutral to Numerical values)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Print classes
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)


# Text Tokenization
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(x)

# Convert text into numerical sequences with integer indices
x_sequences = tokenizer.texts_to_sequences(x)


# Padding the sequences (Ensures all sequences have the same length)
max_length = 100
x_padded = pad_sequences(x_sequences, maxlen=max_length, padding='post')


# Splitting data into Training and Testing sets
x_train, x_test, y_train, y_test = train_test_split(x_padded, y_encoded, test_size=0.2, random_state=42)

print("Data Preparation completed successfully!")
print(f"Vocabulary Size: {len(tokenizer.word_index)}")
print(f"Training Samples: {len(x_train)}, Testing Samples: {len(x_test)}")



# -----------------  Define LSTM model  ---------------

# Define model parameters
embedding_dim = 128
lstm_units = 256
dropout_rate = 0.3


# Build the LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length),

    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(dropout_rate),

    LSTM(lstm_units // 2),   # Second LSTM layer with half units
    Dropout(dropout_rate),

    Dense(128, activation='relu'),    # Fully connected layer
    Dropout(0.2),

    Dense(3, activation='softmax')      # Output Layer
])



# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# ---------  Call backs  -----------------
checkpoint = ModelCheckpoint("best_lstm_model.h5", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# -------------  Train the model  ---------------
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    class_weight=class_weights_dict,
                    callbacks=[checkpoint, early_stop])
# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")