from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class SelfMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads , d_model, dropout):
        super(SelfMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)
        self.drop = Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, attention_mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, attention_mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        output = self.drop(output)
        return output
    
class FeedForward(Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class EncoderLayer(Layer):
    def __init__(self,dim,n_heads,dff,dropout_rate=0.1):
        super(EncoderLayer,self).__init__()
        self.mha = SelfMultiHeadAttention(n_heads,dim,dropout=dropout_rate)
        self.ffn = FeedForward(dim,dff,dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
    def build(self, input_shape):
        super(EncoderLayer, self).build(input_shape)

    def call(self,x,training=True,src_padding_mask=None):
        out1 = self.mha(x,x,x,attention_mask=src_padding_mask,training=training)
        out1 = self.norm1(out1+x)
        out2 = self.ffn(out1)
        out2 = self.norm2(out1+out2)
        return out2
    
class DecoderLayer(Layer):
    def __init__(self,dim,n_heads,dff,dropout_rate=0.1):
        super(DecoderLayer,self).__init__()
        self.mha1 = SelfMultiHeadAttention(n_heads,dim,dropout=dropout_rate)
        self.mha2 = SelfMultiHeadAttention(n_heads,dim,dropout=dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(dim,dff,dropout_rate)
    def build(self, input_shape):
        super(DecoderLayer, self).build(input_shape)

    def call(self,x,enc_output,look_ahead_mask=None,padding_mask=None):
        out1 = self.mha1(x,x,x,attention_mask=look_ahead_mask)
        out1 = self.norm1(x+out1)
        out2 = self.mha2(enc_output,enc_output,out1,attention_mask=padding_mask)
        out2 = self.norm2(out1+out2)
        out3 = self.ffn(out2)
        out3 = self.norm3(out3+out2)
        return out3
    
class Encoder(Layer):
    def __init__(self,dim,n_layers,n_heads,dff,vocab_size,dropout_rate=0.1,length=2048):
        super(Encoder,self).__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.embedding = Embedding(vocab_size,dim)
        self.enc_layers = [
            EncoderLayer(dim,n_heads,dff,dropout_rate) for _ in range(n_layers)
        ]
        self.drop=Dropout(dropout_rate)
        self.pos_encoding = positional_encoding(length,
            self.dim)
        self.dropout=Dropout(dropout_rate)
    def build(self, input_shape):
        super(Encoder, self).build(input_shape)

    def call(self,x,src_padding_mask=None,training=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.n_layers):
            x = self.enc_layers[i](x,src_padding_mask=src_padding_mask)
        x = self.drop(x)
        return x
    
class Decoder(Layer):
    def __init__(self,dim,n_layers,n_heads,dff,vocab_size,dropout_rate=0.1,length=2048):
        super(Decoder,self).__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, dim)
        self.pos_encoding = positional_encoding(length, dim)
        self.dec_layers = [
            DecoderLayer(dim,n_heads,dff,dropout_rate) for _ in range(n_layers)
        ]
        self.dropout = Dropout(dropout_rate)
    def build(self, input_shape):
        super(Decoder, self).build(input_shape)

    def call(self,x,enc_out,look_ahead_mask=None,tgt_padding_mask=None,training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x,training=training)
        for i in range(self.n_layers):
            x = self.dec_layers[i](
                x,
                enc_out,
                look_ahead_mask=look_ahead_mask,
                padding_mask=tgt_padding_mask)
        return x
    
@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self,
                 dim, enc_layers,
                 dec_layers, heads,
                 dff,
                 vocab_size,
                 dropout_rate=0.1,
                 ilen=2048, tlen=2048,**kwargs):
        super(Transformer, self).__init__()
        self.name = ""
        self.dim = dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dff = dff
        self.heads = heads
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.ilen = ilen
        self.tlen = tlen
        self.encoder = Encoder(
            self.dim, self.enc_layers, self.heads,
            self.dff, self.vocab_size, dropout_rate=self.dropout_rate, length=self.ilen)
        self.decoder = Decoder(self.dim, self.dec_layers, self.heads, self.dff,self.vocab_size,
                               dropout_rate=self.dropout_rate, length=self.tlen)
        self.linear = Dense(self.vocab_size)
    
    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({
            'dim': self.encoder.dim,   # Asumiendo que estos son atributos de Encoder y Decoder
            'enc_layers': self.enc_layers,
            'dec_layers': self.dec_layers,
            'heads': self.heads,
            'dff': self.dff,
            'vocab_size': self.vocab_size,
            'dropout_rate': self.dropout_rate,
            'ilen': self.ilen,  # Assuming `length` is the length of the encoder
            'tlen': self.tlen,  # Assuming `length` is the length of the decoder
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self,input_shape):
        super(Transformer,self).build(input_shape)
    
    def call(self, inputs, src_padding_mask=None, look_ahead_mask=None, tgt_padding_mask=None, training=None):
        context, x = inputs
        enc_out = self.encoder(context, src_padding_mask, training=training)
        dec_out = self.decoder(x, enc_out,
                               look_ahead_mask=look_ahead_mask,
                               tgt_padding_mask=tgt_padding_mask,
                               training=training)
        logits = self.linear(dec_out)

        try:
            del logits.keras_mask
        except AttributeError:
            pass
        return logits
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask