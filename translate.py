from preprocess import *

# DEFINE TOKENS DICTIONARY
@st.cache
def get_tokens(dictionary_file='token_dictionary.pkl'):
    with open(f'./models/{dictionary_file}','rb') as f:
        index_word = pickle.load(f)
    word_index = {v:k for k,v in index_word.items()}
    return index_word, word_index

token_dictionary, token_dictionary_index = get_tokens()


### TRANSLATE THE IMAGE
def translate(image, enc, dec):
    max_len = 160
    attention_features_shape = 160
    hidden_units = 256
    attention_plot = np.zeros((max_len, attention_features_shape))

    hidden = tf.zeros((1, hidden_units))
    past_attention = tf.zeros((1, attention_features_shape))

    temp_input = tf.expand_dims(image, 0)

    dec_input = tf.expand_dims([token_dictionary_index['<s>']], 0)
    result = ['<s>']

    features = enc(temp_input, training=False)

    for i in range(max_len):
        predictions, hidden, attention_weights = dec([dec_input,
                                                      features,
                                                      hidden,
                                                      past_attention])

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions, axis=-1)[0].numpy()
        result.append(token_dictionary[predicted_id])

        if token_dictionary[predicted_id] == '<e>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)
        past_attention += attention_weights

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


### PLOT MODEL ATTENTION
def plot_attention(image, result, attention_plot):
    temp_image = np.asarray(image)

    fig = plt.figure(figsize=(15,10))

    len_result = len(result)
    grid_size = int(max(np.ceil(len_result**0.5), 2))
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (10, 16))
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i],fontsize=24)
        ax.set_axis_off()
        img_ax = ax.imshow(temp_image, cmap='gray')
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img_ax.get_extent())

    fig.tight_layout()
    return fig
