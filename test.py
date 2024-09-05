from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import *

f1 = open("./tokenizer.json")

json1 = json.load(f1)

tokenizer = tokenizer_from_json(json1)
batch_size =1
'''
enc_layers=4
dec_layers=4
heads=8
dff=2048
dropout_rate=0.1
ilen=100
tlen=100
'''
dim=512
enc_layers=4
dec_layers=4
heads=8
dff=2048
dropout_rate=0.1
ilen=500
tlen=5000
vocab_size=len(tokenizer.word_index)

begin_token = "<sos>"
end_token = "<eos>"
print("Elaborando modelo...")
transformer = Transformer(
                        dim,
                        enc_layers,
                        dec_layers,
                        heads,
                        dff,
                        vocab_size,
                        dropout_rate=dropout_rate,
                        ilen=ilen,
                        tlen=tlen)

print("Modelo elaborado")
print("Inicializando...")
example_input = (tf.zeros((batch_size, 10)),
                 tf.zeros((batch_size, 100)))
a,b,c = create_masks(example_input[0],example_input[1])
transformer(example_input,a,b,c)


print("Cargando weights")
transformer.load_weights("./wr.weights.h5")


def predict_with_transformer(text, max_len=20):
    text_with_start = f"{begin_token} {text.lower()} {end_token}"
    
    sequence_input = tokenizer.texts_to_sequences([text_with_start])[0]
    print(sequence_input)
    encoder_input = tf.expand_dims(sequence_input,0)
    
    decoder_input = [tokenizer.word_index.get(begin_token, 0)]
    output = tf.expand_dims(decoder_input,0)
    for step in range(max_len):
        # Preparar los inputs para el modelo
        # Crear las máscaras necesarias
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # Obtener logits del modelo
        logits = transformer((encoder_input,output), training=False,
                             src_padding_mask=enc_padding_mask,
                             look_ahead_mask=combined_mask,
                             tgt_padding_mask=dec_padding_mask)

        
        logits_for_last_token = logits[:, -1, :]  # (batch_size, vocab_size)
        #print("Dimensiones de logits_for_last_token:", logits_for_last_token.shape)

        predicted_id = tf.cast([tf.argmax(logits_for_last_token,axis=-1)],tf.int32)
        #print(predicted_id)
        #print(output)
        if predicted_id == tokenizer.word_index.get(end_token, 0):
            return tf.squeeze(output, axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

        print(tokenizer.sequences_to_texts([[tf.squeeze(predicted_id).numpy()]])[0] +" ",end="",flush=True)
        
    # Convertir los tokens a texto utilizando el tokenizer de salida
    predicted_sequence = tokenizer.sequences_to_texts([tf.squeeze(output).numpy()])
        
    # Mostrar la secuencia de tokens generados para depuración
    #print(f"Tokens generados: {decoder_input}")
    return predicted_sequence


while True:
    inp = input(": ")
    pred = predict_with_transformer(inp,1000)
    print()
