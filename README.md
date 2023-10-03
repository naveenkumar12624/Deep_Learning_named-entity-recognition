### EX NO : 05
# <p align="center">Named Entity Recognition</p>

## AIM :

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset :
- We aim to develop an LSTM-based neural network model using Bidirectional Recurrent Neural Networks for recognizing the named entities in the text. 
- The dataset used has a number of sentences, and each words have their tags. 
- We have to vectorize these words using Embedding techniques to train our model.
- Bidirectional Recurrent Neural Networks connect two hidden layers of opposite directions to the same output.
<p align="center">
<img width="350" src="https://user-images.githubusercontent.com/93427237/235588954-f733a2ec-a679-410f-b95c-8cc255aa1939.png">
</p>

</br></br>

## DESIGN STEPS :
### STEP 1:
Import the necessary packages.

### STEP 2:
Read the dataset and fill the null values using forward fill.

### STEP 3:
Create a list of words and tags. Also find the number of unique words and tags in the dataset.

### STEP 4:
Create a dictionary for the words and their Index values. Repeat the same for the tags as well.

### STEP 5:
We done this by padding the sequences and also to acheive the same length of input data.

### STEP 6:
We build the model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.

### STEP 7:
We compile the model to fit the train sets and validation sets.


## PROGRAM :

### Libraries
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras import layers
from keras.models import Model
from tensorflow.keras.preprocessing import sequence
```
### Read & Pre-Process Data
```py
data = pd.read_csv("ner_dataset.csv", encoding="latin1")

data.head(50)

data = data.fillna(method="ffill")

data.head(50)

print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())

print("Unique tags are:", tags)
```
### Define Class to Get Sentance
```py
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences

len(sentences)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

X1 = [[word2idx[w[0]] for w in s] for s in sentences]
```
### Padding
```py
nums = [[1], [2, 3], [4, 5, 6]]
sequence.pad_sequences(nums)

nums = [[1], [2, 3], [4, 5, 6]]
sequence.pad_sequences(nums,maxlen=2)

X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]

y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
```
### LSTM Model
```py
input_word = layers.Input(shape=(max_len,))

embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                                   input_length=max_len)(input_word)
dropout = layers.SpatialDropout1D(0.1)(embedding_layer)

bid_lstm = layers.Bidirectional(
    layers.LSTM(units=100,return_sequences=True,
                recurrent_dropout=0.1))(dropout)

output = layers.TimeDistributed(
    layers.Dense(num_tags,activation="softmax"))(bid_lstm)

model = Model(input_word, output)  

model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x=X_train, y=y_train, validation_data=(X_test,y_test),
    batch_size=50, epochs=3,
)
```
### Metrics
```py
metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()
```
### Prediction
```py
i = 101
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```
## OUTPUT :

| Training Loss, Validation Loss Vs Iteration             | Accuracy, Validation Accuracy Vs Iteration                   |               
| :------------------------------------------------------:| :-----------------------------------------------------------:|
|![image](https://user-images.githubusercontent.com/93427237/235587601-662a7845-93ed-4bcd-a704-3ec10b64bdfe.png)| ![image](https://user-images.githubusercontent.com/93427237/235587615-d2385e35-20b8-484b-924e-e18e056cf433.png) |


### Sample Text Prediction
<p align="center">
<img width="350" src="i1.png">
</p>

## RESULT :
Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.
