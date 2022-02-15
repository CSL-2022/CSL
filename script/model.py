#coding:utf-8
import tensorflow as tf
from utils import *
from tensorflow.python.ops.rnn_cell import GRUCell
from rnn import dynamic_rnn 
class Model(object):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, LEARN_TYPE, Flag="DNN"):
        self.model_flag = Flag
        self.learn_type =LEARN_TYPE
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM], trainable=True)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.cate_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_batch_ph)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_his_batch_ph)            


        with tf.name_scope('init_operation'):    
            self.mid_embedding_placeholder = tf.placeholder(tf.float32,[n_mid, EMBEDDING_DIM], name="mid_emb_ph")
            self.mid_embedding_init = self.mid_embeddings_var.assign(self.mid_embedding_placeholder)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], axis=1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded,self.cate_his_batch_embedded], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1))
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

        short_seq_split = "180:200"
        seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in short_seq_split.split(",")]
        for idx, (left_idx, right_idx) in enumerate(seq_split):
            with tf.name_scope('short_din_layer_{0}'.format(idx)):
                logging.info("short att layer {0}:{1}".format(left_idx, right_idx))
                mask = self.mask[:, left_idx:right_idx]
                attention_output = din_attention(self.item_eb, self.item_his_eb[:, left_idx:right_idx], HIDDEN_SIZE,
                                                 mask, stag='short_att_{0}'.format(idx), return_alphas=False)
                self.short_att_fea = tf.reduce_sum(attention_output, 1)

    def vec_auxiliary_loss(self, query, facts, mask, hidden_size):
        inputs = [self.item_eb]
        attention_output = dinsim_attention(query, facts,
                                         mask=mask, att_func='dot', stag='att_vec_auxiliary')
        att_fea = tf.reduce_sum(attention_output, 1)
        inputs.append(att_fea)

        inp = tf.concat(inputs, -1)
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1_vec')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1_vec')
        dnn1 = prelu(dnn1, scope='dice_1_vec')
        dnn3 = tf.layers.dense(dnn1, 2, activation=None, name='f3_vec')
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        vec_auxiliary_loss = - tf.reduce_mean(tf.log(y_hat) * self.target_ph)
        return vec_auxiliary_loss

    def fcn_net(self, inp):
        print("FCN NET")
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 256, activation=None, name='f1')
        dnn1 = prelu(dnn1, scope='prelu_1')
        dnn2 = tf.layers.dense(dnn1, 256, activation=None, name='f2')
        dnn2 = prelu(dnn2, scope='prelu_2')

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def sal_net(self, inp):
        print("SAL NET")
        self.expert_num_heads = 4
        self.hidden_num_1 = 256
        self.hidden_num_2 = 256
        self.hidden_num_3 = self.hidden_num_2 / self.expert_num_heads
        self.hidden_num_4 = self.hidden_num_3 / self.expert_num_heads
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, self.hidden_num_1, activation=None, name='f1')
        dnn1 = prelu(dnn1, scope='prelu_1')
        dnn2 = tf.layers.dense(dnn1, self.hidden_num_2, activation=None, name='f2')
        dnn2 = prelu(dnn2, scope='prelu_2')
        dnn3_split = tf.concat(tf.split(dnn2, self.expert_num_heads, axis=1), axis=0)
        dnn1_layer_g = prelu(tf.layers.dense(bn1, self.hidden_num_1, activation=None, name='g1'), scope='prelu_3')
        dnn1_layer_g_split = tf.concat(tf.split(dnn1_layer_g, self.expert_num_heads, axis=1), axis=0)
        dnn2_layer_g_split = prelu(tf.layers.dense(dnn1_layer_g_split, self.hidden_num_3, name='g2'), scope='prelu_4')

        dnn4_imul = tf.multiply(dnn3_split, dnn2_layer_g_split)
        dnn4 = tf.concat(tf.split(dnn4_imul, self.expert_num_heads, axis=0), axis=1)
        dnn5 = tf.layers.dense(dnn4, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn5) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def csal_net(self, inp):
        print("CSAL NET")
        # params
        self.tau = 1
        self.expert_num_heads = 4
        self.expert_units = 2
        self.hidden_num_1 = 256
        self.hidden_num_2 = 256
        self.hidden_num_3 = self.hidden_num_2 / self.expert_num_heads
        self.hidden_num_4 = self.hidden_num_3 / self.expert_num_heads
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, self.hidden_num_1, activation=None, name='f1')
        dnn1 = prelu(dnn1, scope='prelu_1')
        dnn2 = tf.layers.dense(dnn1, self.hidden_num_2, activation=None, name='f2')
        dnn2 = prelu(dnn2, scope='prelu_2')
        dnn3_split = tf.concat(tf.split(dnn2, self.expert_num_heads, axis=1), axis=0)
        dnn1_layer_g = prelu(tf.layers.dense(bn1, self.hidden_num_1, activation=None, name='g1'), scope='prelu_3')
        dnn1_layer_g_split = tf.concat(tf.split(dnn1_layer_g, self.expert_num_heads, axis=1), axis=0)
        dnn2_layer_g_split = tf.nn.softmax(tf.layers.dense(dnn1_layer_g_split, self.hidden_num_3, name='g2'))

        dnn4_imul = tf.multiply(dnn3_split, dnn2_layer_g_split)

        dnn4_list = tf.split(dnn4_imul, self.expert_num_heads, axis=0)
        # shard_loss = 0
        output_former = tf.layers.dense(dnn4_list[0], self.expert_units, activation=None, name='former_1')
        # tmp_loss = - tf.reduce_mean(tf.log(tf.nn.softmax(output_former)+ 0.00000001) * self.target_ph)
        # if tmp_loss > self.tau:
            # shard_loss += tmp_loss
        for i in xrange(self.expert_num_heads - 1):
            if i < self.expert_num_heads - 2:
                output_former = tf.layers.dense(tf.concat([output_former, dnn4_list[i + 1]], axis=1), self.expert_units,
                                                activation=None)
            else:
                output_former = tf.layers.dense(tf.concat([output_former, dnn4_list[i + 1]], axis=1), 2,
                                                activation=None)
            # tmp_loss = - tf.reduce_mean(tf.log(tf.nn.softmax(output_former)+ 0.00000001) * self.target_ph)
            # if tmp_loss > self.tau:
                # shard_loss += tmp_loss
        # dnn5 = tf.layers.dense(dnn4, 2, activation=None, name='f3')
        dnn5 = output_former
        # dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn5) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
                        # + shard_loss
            # self.loss =  shard_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def init_uid_weight(self, sess, uid_weight):
        sess.run(self.uid_embedding_init,feed_dict={self.uid_embedding_placeholder: uid_weight})
    
    def init_mid_weight(self, sess, mid_weight):
        sess.run([self.mid_embedding_init],feed_dict={self.mid_embedding_placeholder: mid_weight})

    def save_mid_embedding_weight(self, sess):
        embedding = sess.run(self.mid_embeddings_var)
        return embedding

    def save_uid_embedding_weight(self, sess):
        embedding = sess.run(self.uid_bp_memory)
        return embedding                                 
    
    def train(self, sess, inps):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cate_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cate_his_batch_ph: inps[4],
            self.mask: inps[7],
            self.target_ph: inps[8],
            self.lr: inps[9]
        })
        aux_loss = 0
        return loss, accuracy, aux_loss            

    def calculate(self, sess, inps):
        probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cate_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cate_his_batch_ph: inps[4],
            self.mask: inps[7],
            self.target_ph: inps[8]
        })
        aux_loss = 0
        return probs, loss, accuracy, aux_loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_DNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256, LEARN_TYPE='FCN'):
        super(Model_DNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, LEARN_TYPE, Flag="DNN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        if self.learn_type=='CSAL':
            self.csal_net(inp)
        elif  self.learn_type=='SAL':
            self.sal_net(inp)
        else:
            self.fcn_net(inp)


class Model_DIN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256, LEARN_TYPE='FCN'):
        super(Model_DIN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, LEARN_TYPE, Flag="DIN")
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, HIDDEN_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, att_fea], -1)
        if self.learn_type=='CSAL':
            self.csal_net(inp)
        elif  self.learn_type=='SAL':
            self.sal_net(inp)
        else:
            self.fcn_net(inp)


class Model_DIEN(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=400, LEARN_TYPE='FCN'):
        super(Model_DIEN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE,
                                         BATCH_SIZE, SEQ_LEN, LEARN_TYPE, Flag="DIEN")

        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, _ = dynamic_rnn(GRUCell(2 * EMBEDDING_DIM), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, mask=self.mask, mode="LIST",
                                                return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.sequence_length, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1)
        if self.learn_type=='CSAL':
            self.csal_net(inp)
        elif  self.learn_type=='SAL':
            self.sal_net(inp)
        else:
            self.fcn_net(inp)


class Model_SIM(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256, LEARN_TYPE='FCN', args={}):
        super(Model_SIM, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE,
                                        BATCH_SIZE, SEQ_LEN, LEARN_TYPE, Flag="SIM")

        inputs = [self.item_eb, self.item_his_eb_sum]

        args['long_seq_split'] = '0:180'
        args['first_att_top_k'] = 100
        args['use_first_att'] = True
        args['att_func'] = 'dot'
        args['level'] = 'train'

        seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args['long_seq_split'].split(",")]
        for idx, (left_idx, right_idx) in enumerate(seq_split):
            with tf.name_scope('long_att_layer_{0}'.format(idx)):
                mask = self.mask[:, left_idx:right_idx]
                self.vec_loss = self.vec_auxiliary_loss(self.item_eb, self.item_his_eb[:, left_idx:right_idx], mask,
                                                        HIDDEN_SIZE)

                attention_output, scores = dinsim_attention(self.item_eb, self.item_his_eb[:, left_idx:right_idx],
                                                         mask=self.mask[:, left_idx:right_idx], att_func='dot',
                                                         return_alphas=True,
                                                         stag='att_vec_{0}'.format(idx))
                top_k = args['first_att_top_k']
                scores -= top_kth_iterative(scores, top_k)
                if args['level'].lower() == 'debug':
                    scores = tf.Print(scores, ["score:", scores[0]], summarize=1000)
                if args['use_first_att']:
                    mask = tf.cast(tf.greater(scores, tf.zeros_like(scores)), tf.float32)
                    if args['level'].lower() == 'debug':
                        mask = tf.Print(mask, ["mask:", mask[0]], summarize=1000)
                att_func = args['att_func']
                attention_output, scores = dinsim_attention(self.item_eb, self.item_his_eb[:, left_idx:right_idx],
                                                         HIDDEN_SIZE, mask, att_func=att_func,
                                                         stag='att_{0}'.format(idx), return_alphas=True)
                self.att_scores = scores
                att_fea = tf.reduce_sum(attention_output, 1)
                inputs.append(att_fea)

                item_his_sum_emb = tf.reduce_sum(self.item_his_eb[:, left_idx:right_idx] * mask[:, :, None], 1) / (
                            tf.reduce_sum(mask, 1, keepdims=True) + 1.0)
                inputs.append(item_his_sum_emb)
                inputs.append(self.short_att_fea)

        logging.info(inputs)
        inp = tf.concat(inputs, 1)

        if self.learn_type=='CSAL':
            self.csal_net(inp)
        elif  self.learn_type=='SAL':
            self.sal_net(inp)
        else:
            self.fcn_net(inp)
