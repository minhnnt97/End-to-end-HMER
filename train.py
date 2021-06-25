from architecture import *


#########################
##### TRAINING STEP #####
#########################
@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the labels are not related from image to image
    dec_input      = tf.expand_dims([tokenizer.word_index['<s>']] * target.shape[0], 1)
    hidden         = tf.zeros((target.shape[0], units))
    past_attention = tf.zeros((target.shape[0], 160)) # (batch_size, 160)


    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, att_W = decoder([dec_input, features, hidden, past_attention])

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

            # update past attention
            past_attention += att_W
        
        loss += tf.reduce_sum(encoder.losses) + tf.reduce_sum(decoder.losses)

    total_loss = (loss / int(target.shape[1]))


    # BACKPROP
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    gradients,_ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss



###########################
##### VALIDATION STEP #####
###########################
@tf.function
def test_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    dec_input      = tf.expand_dims([tokenizer.word_index['<s>']] * target.shape[0], 1)
    hidden         = tf.zeros((target.shape[0], units))#decoder.reset_state(batch_size=target.shape[0])
    past_attention = tf.zeros((target.shape[0], 160)) # (batch_size, 160)

    features = encoder(img_tensor, training=False)

    for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, hidden, att_W = decoder([dec_input, features, hidden, past_attention])

        loss += loss_function(target[:, i], predictions)

        # use teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)#tf.expand_dims(tf.argmax(predictions, axis=-1), axis=-1)

        # update past attention
        past_attention += att_W
    
    loss += tf.reduce_sum(encoder.losses) + tf.reduce_sum(decoder.losses)

    total_loss = (loss / int(target.shape[1]))

    return total_loss




##################
##### METRIC #####
##################
def cmp_result(label,rec):
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)


def get_metrics(enc, dec, img_path_array, lbl_array):
    total_expr = 0
    total_dist = 0
    total_llen = 0
    len_test = len(img_path_array)
    for idx in range(len_test):
        image = img_path_array[idx]
        true_label = [i for i in lbl_array[idx] if i not in [0]][1:-1]

        result, _ = predict(image, enc, dec)
        pred_label = [tokenizer.word_index[r] for r in result][1:-1]
        dist, llen = cmp_result(true_label, pred_label)
        total_dist += dist
        total_llen += llen
        if dist==0:
            total_expr += 1

    expr_rate = total_expr / len_test
    total_wer = total_dist / total_llen
    return expr_rate, total_wer
