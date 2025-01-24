import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("FinalDataset.csv")

# Tokenize the 'Code' column
tokenizer = Tokenizer(num_words=10000)  # Set the maximum number of tokens
tokenizer.fit_on_texts(df['Code'])
X = tokenizer.texts_to_sequences(df['Code'])
X = pad_sequences(X, padding='post', maxlen=100)  # Pad sequences to have equal length

# Encode the 'Label' column (True -> 1, False -> 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Label'])

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)