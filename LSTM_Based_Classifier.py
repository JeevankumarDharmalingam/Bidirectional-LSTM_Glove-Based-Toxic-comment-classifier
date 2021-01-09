import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM,Dense,Flatten,Dropout,Embedding,Bidirectional,concatenate,Dense,Input,LSTM,SpatialDropout1D,Bidirectional,Activation,Conv1D,GRU,GlobalAveragePooling1D,GlobalMaxPooling1D
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./train.csv")

train_text = train_df["comment_text"]

train_target = train_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

max_feature = 10000
max_len = 512
embed_size = 300

tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts(train_text)
X = tokenizer.texts_to_sequences(texts=train_text)
X = pad_sequences(X,maxlen=max_len)

EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

vocab_size = len(tokenizer.word_index)+1

embedding_matrix = np.zeros((vocab_size,embed_size))
for word,i in tqdm(tokenizer.word_index.items()):
    embedding_value = embeddings_index.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value



#Creating Network @Jeevankumar
model = Sequential()
model.add(Embedding(vocab_size,embed_size,weights=[embedding_matrix],trainable = False,input_length=max_len))
model.add(Bidirectional(LSTM(300,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)))
model.add( GlobalMaxPooling1D())
model.add(Dense(1028,activation = 'relu'))
model.add(Dense(6,activation = 'sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])


model.summary()


X_train,X_val,Y_train,Y_val = train_test_split(X,train_target,test_size = 0.25,random_state = 42)

filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=10)
callback = [checkpoint,early]


model.fit(X_train, Y_train, batch_size=256,
          epochs=5, validation_data=(X_val, Y_val),callbacks = callback,verbose=1)