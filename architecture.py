import tensorflow as tf
import tensor2tensor as t2t

from tensorflow.keras.regularizers import l2

##############################
########## ENCODER ###########
##############################
class CNN_Encoder(tf.keras.Model):
    '''
    The custom CNN encoder.
    Extracts features from an image tensor and 
    returns the positional embeddings of those features.

    Call arguments:
    x : a 4D tensor of shape (batch_size, height, width, channels)

    training: a boolean to indicate whether the model is in inference mode (training=False)
              or training mode (training=True). Default is True

    Output:
    out : a 3D tensor of shape(batch_size, HxW, n_filters)
          where H,W, n_filters are the dimenions of the final block of feature maps.
    '''
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.n_filters_list = [32, 64, 64, 128, 128]
        self.dropout_list   = [0., 0., 0., 0.2, 0.2]
        self.n_conv = 4
        self.cnn = tf.keras.Sequential()
        for n_filters, dropout in zip(self.n_filters_list, self.dropout_list):
            for i in range(self.n_conv):
                self.cnn.add(tf.keras.layers.Conv2D(n_filters, 3, padding='same', kernel_regularizer=l2(1e-4)))
                self.cnn.add(tf.keras.layers.BatchNormalization())
                self.cnn.add(tf.keras.layers.Activation('relu'))
                if dropout>0 and i<self.n_conv-1:
                    self.cnn.add(tf.keras.layers.Dropout(dropout))
            self.cnn.add(tf.keras.layers.MaxPool2D(2,2, padding='same'))


    def call(self, x, training=True):
        x = self.cnn(x, training=training)
        x = t2t.layers.common_attention.add_timing_signal_nd(x)
        H,W = x.shape[1:3]
        out = tf.reshape(x, [-1, H*W, self.n_filters_list[-1]])

        return out




##############################
######### ATTENTION  #########
##############################
class CoverageAttention(tf.keras.Model):
    '''
    A custom attention mechanism that tries to follow the paper
    "Watch, Attend, Parse: An end-to-end neural network based
    approach to handwritten mathematical expression recognition".

    This mechanism is based on the attention mechanism by Bahdanau.
    It calculates the attention score for the features using the 
    features and the previous hidden state of the RNN (like Bahdanau's
    attention), as well as a novel element that keeps track of the
    past attention scores previously calculated by the RNN.

    Call arguments:
    x : a list of 3 elements [features, hidden, past_attention]

    features       : a tensor of shape (batch_size, feature_len, feature_maps)
    hidden         : a tensor of shape (batch_size, hidden_units)
    past_attention : a tensor of shape (batch_size, feature_len)

    Outputs:
    context_vector    : a tensor of shape (batch_size, feature_maps)
    attention_weights : a tensor of shape (batch_size, feature_len, 1)
    '''
    def __init__(self, units):
        super(CoverageAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, kernel_regularizer=l2(1e-4))
        self.W2 = tf.keras.layers.Dense(units, kernel_regularizer=l2(1e-4))
        self.Uf = tf.keras.layers.Dense(units, kernel_regularizer=l2(1e-4))
        self.V = tf.keras.layers.Dense(1, kernel_regularizer=l2(1e-4))


    def call(self, x):
        features, hidden, past_attention = x
        # features(CNN_encoder output) shape == (batch_size, feature_len, feature_maps)
        # past_attention shape == (batch_size, feature_len)

        # hidden shape == (batch_size, hidden_units)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        past_attention = tf.expand_dims(past_attention, -1)

        # attention_hidden_layer shape == (batch_size, feature_len, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis) +
                                             self.Uf(past_attention)))

        # score shape == (batch_size, feature_len, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, feature_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_units)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights




##############################
########## DECODER  ##########
##############################
class RNN_Decoder(tf.keras.Model):
    '''
    The custom RNN decoder.
    This decoder is based on the Tensorflow tutorial on image captioning.

    The decoder takes in a list of input tensors and compute the
    logits for single token predictions. Uses the CoverageAttention
    model implemented above to calculate the context vector and the
    attention scores.

    The context vector will be appended to the embedding vector of the
    input token and passed into the RNN layer. The RNN layer will then
    output a token tensor and the RNN hidden state. The token tensor
    will then pass through 2 dense layers, with the last one giving out
    the logits for the predicted token.

    Initialize arguments:
    embedding_dim : (int) the dimension of the word embeddings
    units         : (int) the number of units of the hidden state of the 
                    RNN layer
    vocab_size    : (int) the size of vocabulary, used for word embedding
                    and final classification

    Call arguments:
    inputs : a list of 4 elements [x, features, hidden, past_attention]

    x              : a tensor of shape (batch_size, 1)
                     The token index of the input token.
                     The RNN will use this combined with the context vector
                     produced by the attention layer for predicting the next token.
    features       : a tensor of shape (batch_size, feature_len, feature_maps)
                     Used in the attention layer.
    hidden         : a tensor of shape (batch_size, units)
                     The previous hidden state of the RNN layer. Used in the
                     attention layer.
    past_attention : a tensor of shape (batch_size, feature_len)
                     The past attention calculated throughout the sequence. Used
                     in the attention layer.

    Outputs:
    x                 : a tensor of shape (batch_size, vocab_size)
                        The logits of the predicted token.
    state             : a tensor of shape (batch_size, units)
                        The hidden state of the RNN layer.
    attention_weights : a tensor of shape (batch_size, feature_len)
                        The attention scores of this call.
    '''
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = CoverageAttention(self.units)#MultiHeadAttention(5, key_dim=128)

    def call(self, inputs):
        x, features, hidden, past_attention = inputs

        # defining attention as a separate model
        context_vector, attention_weights = self.attention([features, hidden, past_attention])

        # x.shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x.shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        # output.shape == (batch_size, 1, embedding_dim + hidden_size)
        # state.shape  == (batch_size, hidden_size)
        output, state = self.gru(x)  

        # shape == (batch_size, 1, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size, vocab_size)
        x = self.fc2(x)

        return x, state, tf.reshape(attention_weights, [-1,attention_weights.shape[1]])

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
