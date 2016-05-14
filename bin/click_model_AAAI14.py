import os
import sys
import math
import numpy as np
import tflearn
import tensorflow as tf


position_num = 10
add_feature_num = 6

def arr_string(arr, sep="\t"):
    info = ""
    if len(arr) > 0:
        info += str(arr[0])
        for i in range(1, len(arr)):
            info += sep + str(arr[i])
    return info

def save_predict(labels, preds, file_name):
    out_file = open(file_name, "w")
    for i in range(0, len(preds)):
        out_file.write(arr_string(labels[i]) + "\t" + arr_string(preds[i]) + "\n")
    out_file.close()

def print_predict(labels, preds, num=100):
    if num < 0:
        num = len(preds)
    for i in range(0, num):
        ret_str = str(labels[i]) + "\n" + str(preds[i])
        print ret_str

def perplexity_score(labels, preds):
        positionPerplexity = [0.0] * position_num
        positionPerplexityClickSkip = [[0.0, 0.0] for i in xrange(position_num)]
        counts = [0] * position_num
        countsClickSkip = [[0, 0] for i in xrange(position_num)]
        for label, pred in zip(labels, preds):
            for i in range(0, len(label)):
                click = 1 if label[i] else 0
                tmp_pred = max(min(pred[i], 0.99999), 0.00001)
                logProb = math.log(tmp_pred, 2)
                if click == 0:
                    logProb = math.log(1 - tmp_pred, 2)
                positionPerplexity[i] += logProb
                positionPerplexityClickSkip[i][click] += logProb
                counts[i] += 1
                countsClickSkip[i][click] += 1
        positionPerplexity = [2 ** (-x / count if count else x) for (x, count) in zip(positionPerplexity, counts)]
        positionPerplexityClickSkip = [[2 ** (-x[click] / (count[click] if count[click] else 1) if count else x) \
                for (x, count) in zip(positionPerplexityClickSkip, countsClickSkip)] for click in xrange(2)]
        perplexity = sum(positionPerplexity) / len(positionPerplexity)
        ret_str = "---------\n"
        ret_str += "Perplexity\t" + str(perplexity) + "\n"
        ret_str += "positionPerplexity"
        for i in range(0, position_num):
            ret_str += "\t" + str(positionPerplexity[i])
        ret_str += "\n"

        ret_str += "positionPerplexitySkip"
        for i in range(0, position_num):
            ret_str += "\t" + str(positionPerplexityClickSkip[0][i])
        ret_str += "\n"

        ret_str += "positionPerplexityClick"
        for i in range(0, position_num):
            ret_str += "\t" + str(positionPerplexityClickSkip[1][i])
        ret_str += "\n------------\n"
        #print ret_str
        return perplexity, ret_str

def get_average_feature(id_list, vocab):
    ndim = vocab.shape[1]
    sum_val = np.zeros(ndim)
    for i in range(0, len(id_list)):
        cid = int(id_list[i])
        sum_val += vocab[cid]
    sum_val /= len(id_list)
    return sum_val

def form_xy(q, a, y, add, qids, vocab):
    data_size = len(y)
    x = []
    for i in range(0, data_size):
        tmp_feature = []
        #add query feature
        q_feature = get_average_feature(q[i], vocab)
        q_feature = np.hstack([q_feature, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tmp_feature.append(q_feature)
        #add result feature
        for j in range(0, position_num):
            ai_feature = get_average_feature(a[i][j], vocab)
            add_feature = []
            for k in range(1, add_feature_num + 1):
                add_feature.append(add[i][1 + k * position_num + j])
            ai_feature = np.hstack([ai_feature, np.array(add_feature)])
            tmp_feature.append(ai_feature)
        x.append(np.array(tmp_feature))
    return np.array(x), y

def batch_predict(model, inputX, batch_size=500):
    ret = []
    total_count = len(inputX)
    count_index = 0
    while count_index * batch_size < total_count:
        print "Batch Predict " + str(count_index * batch_size) + " / " + str(total_count)
        tmpX = inputX[count_index * batch_size: min((count_index + 1) * batch_size, total_count)]
        tmpY = model.predict(tmpX)
        for i in range(0, len(tmpY)):
            ret.append(tmpY[i])
        count_index += 1
    return ret

def main():
    mode = 'k_time_data1'.upper()

    print "Running training in the {} setting".format(mode)
    data_dir = mode

    add_train = np.load(os.path.join(data_dir, 'train.additions.npy'))
    q_train = np.load(os.path.join(data_dir, 'train.questions.npy'))
    a_train = np.load(os.path.join(data_dir, 'train.answers.npy'))
    y_train = np.load(os.path.join(data_dir, 'train.labels.npy'))
    qids_train = np.load(os.path.join(data_dir, 'dev.qids.npy'))

    add_dev = np.load(os.path.join(data_dir, 'dev.additions.npy'))
    q_dev = np.load(os.path.join(data_dir, 'dev.questions.npy'))
    a_dev = np.load(os.path.join(data_dir, 'dev.answers.npy'))
    y_dev = np.load(os.path.join(data_dir, 'dev.labels.npy'))
    qids_dev = np.load(os.path.join(data_dir, 'dev.qids.npy'))

    add_test = np.load(os.path.join(data_dir, 'test.additions.npy'))
    q_test = np.load(os.path.join(data_dir, 'test.questions.npy'))
    a_test = np.load(os.path.join(data_dir, 'test.answers.npy'))
    y_test = np.load(os.path.join(data_dir, 'test.labels.npy'))
    qids_test = np.load(os.path.join(data_dir, 'test.qids.npy'))

    # Load word2vec embeddings
    fname = os.path.join(data_dir, 'emb_vectors.skip.1124.4m.10w.npy')

    print "Loading word embeddings from", fname
    vocab_emb = np.load(fname)
    print "word embedding shape: " + str(vocab_emb.shape)

    #(vocabulary_size, embedding_size) = vocab_emb.shape
    #embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    #embed = tf.nn.embedding_lookup(embeddings, train_inputs)


    X_train, Y_train = form_xy(q_train, a_train, y_train, add_train, qids_train, vocab_emb)
    X_dev, Y_dev = form_xy(q_dev, a_dev, y_dev, add_dev, qids_dev, vocab_emb)
    X_test, Y_test = form_xy(q_test, a_test, y_test, add_test, qids_test, vocab_emb)

    print "train size " + str(X_train.shape)
    print "dev size " + str(X_dev.shape)
    print "test size " + str(X_test.shape)

    model_name = "./" + mode + "-rnn.model"
    seq_dim = position_num + 1
    feature_dim = vocab_emb.shape[1] + add_feature_num
    net = tflearn.input_data(shape=[None, seq_dim, feature_dim])
    #net = tflearn.lstm(net, 32, return_seq=True)
    net = tflearn.simple_rnn(net, 512)
    net = tflearn.fully_connected(net, 512, activation='tanh')
    net = tflearn.fully_connected(net, position_num, activation='sigmoid')
    net = tflearn.regression(net, optimizer='SGD', loss='categorical_crossentropy', name="output1")
    model = tflearn.DNN(net, tensorboard_verbose=3)

    #train model
    model.fit(X_train, Y_train, n_epoch=15, validation_set=(X_dev, Y_dev), show_metric=True,snapshot_step=1000)
    model.save(model_name)
    #model.load(model_name)

    #test model
    predict_Y_test = batch_predict(model, X_test)
    perp, perp_str = perplexity_score(y_test, predict_Y_test)
    print perp_str

    output_file_name = "./" + mode + "-rnn.predict.result"
    save_predict(y_test, predict_Y_test, output_file_name)
    #print_predict(y_test, predict_Y_test)

if __name__ == '__main__':
    main()