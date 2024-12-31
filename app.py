import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

model=load_model('simple_rnn_model.h5')

def decode_review(text):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in text])

def preprocessing_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=pad_sequences([encoded_review],maxlen=250)
    return padded_review

def predict(review):
  preprocessed_review=preprocessing_text(review)
  prediction=model.predict(preprocessed_review)
  if prediction>0.5:
    return 'POSITIVE' ,prediction[0][0]
  else:
    return 'NEGATIVE' ,prediction[0][0]

op=predict(review)

st.title('BOOK MY TICKETS')

movie= st.selectbox('Please select the movie',['UI','Robert','MAX','Martin','PUSHPA'])

message = st.text_area('Enter your message about the movie:')

star_options = [1, 2, 3, 4, 5]  

stars = st.selectbox('Please rate the movie:', star_options)

if not message:  # Check if message is empty
    if stars > 2:
        op = 'POSITIVE'
    else:
        op = 'NEGATIVE'
        
st.write(f"You selected: {movie}")
st.write(f"Your message: {message}")
st.write(f"Your stars: {stars}")
st.write(f"Your review: {op[0]}")
st.write(f"Your rating: {op[1]}")










