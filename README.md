# Urdu Sentiment Analysis using BiLSTM

This repository contains code for training and evaluating a BiLSTM model for Urdu sentiment analysis. The model is trained on a dataset of IMDb Urdu movie reviews.

## Dataset

The dataset used in this project consists of IMDb Urdu movie reviews. The training set (`imdb_urdu_reviews_train.csv`) and the testing set (`imdb_urdu_reviews_test.csv`) are provided in the `data` directory.

## Installation

1. Clone the repository: git clone https://github.com/BasharSiddiqui/urdu-sentiment-analysis.git
2. Change into the project directory: cd urdu-sentiment-analysis
## Usage

1. Data Preparation:
- Unzip the dataset archive file (`archive (3).zip`) located in the `data` directory.
- Run the following code to load and preprocess the data:
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  
  # Load the dataset
  train = pd.read_csv('data/imdb_urdu_reviews_train.csv')
  test = pd.read_csv('data/imdb_urdu_reviews_test.csv')
  
  # Combine the training and testing sets
  data = pd.concat([train, test]).reset_index(drop=True)
  
  # Encode the sentiment labels
  le = LabelEncoder()
  le.fit(data['sentiment'])
  data['encoded_sentiments'] = le.transform(data['sentiment'])
  
  # Tokenize the text data
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(data['review'].tolist())
  word_index = tokenizer.word_index
  vocab_size = len(word_index) + 1
  sequences = tokenizer.texts_to_sequences(data['review'].tolist())
  
  # Pad sequences
  max_length = max([len(seq) for seq in sequences])
  padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
  
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['encoded_sentiments'], test_size=0.2, random_state=42)
  ```

2. Training and Evaluation:
- Run the following code to train and evaluate the BiLSTM model:
  ```python
  import tensorflow as tf
  from sklearn.metrics import f1_score, precision_score
  import matplotlib.pyplot as plt
  
  # Define the BiLSTM model architecture
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(2, activation='softmax')
  ])
  
  # Compile the model
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  # Train the model
  history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
  
  # Evaluate the model
  y_pred = model.predict(X_test)
  y_pred = np.argmax(y_pred, axis=1)
  
  f1 = f1_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  validation_accuracy = history.history['val_accuracy'][-1]
  ```
  
3. Plotting the Results:
- Run the following code to plot the accuracy and loss curves:
  ```python
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()
  
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train', 'Validation'], loc='upper right')
  plt.show()
  ```

4. Model Saving:
- Run the following code to save the trained model:
  ```python
  model.save("my_model.h5")
  ```

## Results

The trained model achieved the following metrics on the validation set:
- F1 Score: 0.87
- Precision: 0.83
- Validation Accuracy: 87%
