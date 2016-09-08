import os, sys, time
import numpy as np
import tensorflow as tf
import datagen as dg

class Config(object):
    """ The model hyperparams and data information

    """
    def __init__(self,
            config_list=[300, 64, 1000, 1024, 15, 0.9, 0.00002, 0.001, 25]):
        [self.embed_size,
         self.batch_size,
         self.label_size,
         self.hidden_size,
         self.max_epochs,
         self.dropout,
         self.lr,
         self.l2,
         self.max_ques_len] = config_list

class CNN_LSTM_VQAModel():
    """ Visual Question Answering using CNN and LSTM model.

    """

    def load_data(self, debug=False):
        """ load the image, question, answer data from dataset """
        self.train_set, self.test_set,self.vocab, self.ans_lk_table,\
         = dg.build_voc_and_get_data(self.config.max_ques_len)
        self.wv, _, _ = dg.word_embed(self.vocab, 'glove.6B.300d')
        # Create one-hot table for labels
        label_size = len(self.ans_lk_table)
        assert label_size == self.config.label_size
        for i in range(len(self.train_set)):
            ans_id = self.train_set[i][2]
            ans_vec = np.zeros(label_size)
            ans_vec[ans_id] = 1
            self.train_set[i][2] = ans_vec
        if debug:
            self.valid_set = self.train_set[1024: 2048]
            self.train_set = self.train_set[:1024]
            self.test_set = self.test_set[:65]
        else:
            self.valid_set = self.train_set[:len(self.train_set) // 4]
            self.train_set = self.train_set[len(self.train_set) // 4:]

    def add_placeholders(self):
        """ placeholder tensors for input and labels

            input_placeholder: input tensor of shape
                                (None, max_ques_len), type tf.int32
            label_placeholder: label tensor of shape
                                (batch_size, label_size), type tf.float32
            dropout_placeholder: Dropout value placeholder (scalar)
                                type tf.float32
        """
        self.input_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.max_ques_len], name="Input")
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=[self.config.batch_size, self.config.label_size],
                             name="Target")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="Dropout")

    def add_embedding(self, wv=None):
        with tf.device('/cpu:0'):
            if wv is not None:
                embedding = tf.get_variable(name="Embed",
                    shape=wv.shape, initializer=tf.constant_initializer(wv),
                                            trainable=False)
            else:
                embedding = tf.get_variable(name="Embed",
                    shape=[len(self.vocab), self.config.embed_size], trainable=True)
            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            tmp = tf.split(1, self.config.max_ques_len, inputs)
            inputs = [tf.squeeze(x, [1]) for x in tmp]
        return inputs

    def add_projection(self, output, cnn_outputs=None):
       with tf.variable_scope('Projection'):
           U = tf.get_variable(
               'U', [self.config.hidden_size, self.config.label_size])
           b = tf.get_variable(
               'b', [self.config.label_size])
           output = tf.matmul(output, U) + b
       return output

    def add_lstm_model(self, inputs):
        """ LSTM model for Question Encoding.

            inputs: list (length=max_ques_len) of input tensor of shape
                        (batch_size, embed_size), type tf.float32

            lstm_output: final state of lstm, tensor of shape
                        (batch_size, hidden_size), type tf.float32
        """
        batch_size, hidden_size, embed_size = \
            self.config.batch_size, self.config.hidden_size,\
            self.config.embed_size

        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]

        with tf.variable_scope('LSTM') as scope:
            self.initial_state = tf.zeros([batch_size, hidden_size])
            ht = self.initial_state
            Ct = ht
            for step, xt in enumerate(inputs):
                if step > 0:
                    scope.reuse_variables()
                # forget gate
                Wf = tf.get_variable("Wf", [embed_size, hidden_size])
                Uf = tf.get_variable("Uf", [hidden_size, hidden_size])
                bf = tf.get_variable("bf", [hidden_size])
                ft = tf.nn.sigmoid(tf.matmul(xt, Wf) + tf.matmul(ht, Uf) + bf)
                # input gate
                Wi = tf.get_variable("Wi", [embed_size, hidden_size])
                Ui = tf.get_variable("Ui", [hidden_size, hidden_size])
                bi = tf.get_variable("bi", [hidden_size])
                it = tf.nn.sigmoid(tf.matmul(xt, Wi) + tf.matmul(ht, Ui) + bi)
                Wc = tf.get_variable("Wc", [embed_size, hidden_size])
                Uc = tf.get_variable("Uc", [hidden_size, hidden_size])
                bc = tf.get_variable("bc", [hidden_size])
                _Ct = tf.nn.tanh(tf.matmul(xt, Wc) + tf.matmul(ht, Uc) + bc)
                # cell
                Ct = ft * Ct + it * _Ct
                #output
                Wo = tf.get_variable("Wo", [embed_size, hidden_size])
                Uo = tf.get_variable("Uo", [hidden_size, hidden_size])
                bo = tf.get_variable("bo", [hidden_size])
                ot = tf.nn.sigmoid(tf.matmul(xt, Wo) + tf.matmul(ht, Uo) + bo)
                ht = ot * tf.nn.tanh(Ct)
            self.final_state = ht
        with tf.variable_scope("LSTMDropout"):
            lstm_output = tf.nn.dropout(self.final_state, self.dropout_placeholder)
        return lstm_output

    def add_loss_op(self, y):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, self.labels_placeholder))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        #self.load_data(debug=True)
        self.add_placeholders()
        self.inputs = self.add_embedding(self.wv)
        self.lstm_output = self.add_lstm_model(self.inputs)
        self.output = self.add_projection(self.lstm_output)
        #self.prediction = self.predict(self.output)
        self.loss = self.add_loss_op(self.output)
        self.predictions = tf.nn.softmax(self.output)
        one_hot_prediction = tf.argmax(self.predictions, 1)
        correct_predictions = tf.equal(
            tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(
            tf.cast(correct_predictions, 'int32'))
        self.train_step = self.add_training_op(self.loss)

    def run_epoch(self, session, input_data, train_op=None):
        config = self.config
        dropout = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dropout = 1
        total_steps = len(input_data) // self.config.batch_size
        total_loss = []
        for step, (x, y) in enumerate(
            dg.data_iter(input_data, config.batch_size)):
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.dropout_placeholder: dropout}
            loss, _ = session.run(
                [self.loss, train_op], feed_dict=feed)
            total_loss.append(loss)
            if step % 10 == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, loss))
                sys.stdout.flush()
        print('\n')
        return np.mean(total_loss)

    def predict(self, session, data):
        """ Prediction generator """
        dropout = 1
        pred_res = []
        confi_res = []
        total_steps = len(data) // self.config.batch_size
        for step, (x, y) in enumerate(
            dg.data_iter(data, self.config.batch_size)):
            if x.shape[0] < self.config.batch_size:
                break
            feed = {self.input_placeholder: x,
                self.dropout_placeholder: dropout}
            preds = session.run(self.predictions, feed_dict=feed)
            predicted_indices = preds.argmax(axis=1)
            pred_res.extend(predicted_indices)
            confident = preds.max(axis=1)
            confi_res.extend(confident)
            if step % 10 == 0:
                sys.stdout.write('\r{} / {}'.format(
                    step, total_steps))
                sys.stdout.flush()
        print("\n")

        return pred_res, confi_res


def test_VQA():
    config = Config()

    with tf.variable_scope("CNN_LSTM_VQA") as scope:
        model = CNN_LSTM_VQAModel(config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_loss = float('inf')
        best_val_epoch = 0

        session.run(init)
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
            train_loss = model.run_epoch(
                session, model.train_set, model.train_step)
            valid_loss = model.run_epoch(session, model.valid_set)
            print("Training loss: {}".format(train_loss))
            print("Validation loss: {}".format(valid_loss))
            print("Total time: {}".format(time.time() - start))
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(session, './weights/Vqa.weights')

        saver.restore(session, './weights/Vqa.weights')
        print("Test Model: ")
        pred_indices, _ = model.predict(session, model.test_set)
        test_ans_list = [x[2] for x in model.test_set]
        num = len(test_ans_list) - len(test_ans_list) % config.batch_size
        test_ans_list = test_ans_list[:num]
        accuracy = np.equal(pred_indices, test_ans_list).sum() / \
            len(pred_indices)
        print("accuracy: {}".format(accuracy))

if __name__ == "__main__":
    test_VQA()
