import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import  _Linear
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs


def din_attention(query, facts, attention_size, mask=None, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("query_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all

    if mask is not None:
        mask = tf.equal(mask, tf.ones_like(mask))
        key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    
    if return_alphas:
        return output, scores
        
    return output


def dinsim_attention(query, facts, attention_size=None, mask=None, stag='null', mode='SUM', att_func='all',softmax_stag=1, time_major=False, return_alphas=False,reuse=None,top_k=None):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("query_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    logging.info(att_func)
    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    if att_func == 'all':
        din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag,reuse=reuse)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag,reuse=reuse)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag,reuse=reuse)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all

    if att_func == 'dot':
        din_all = tf.reduce_sum(queries * facts,axis=-1)
        scores = tf.reshape(din_all,[-1,1,tf.shape(facts)[1]])


    # scores = tf.Print(scores,["score1",scores[0]],summarize=1000)
    if mask is not None:
        mask = tf.equal(mask, tf.ones_like(mask))
        key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    # scores = tf.Print(scores,["score2",key_masks[0],scores[0]],summarize=1000)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]
        # scores = tf.Print(scores, ["score3",  scores[0]], summarize=1000)
        # scores = scores[:,0,:] * tf.cast(mask,tf.float32)
        # scores = scores[:,None,:]
        # scores = tf.Print(scores, ["score4",  scores[0]], summarize=1000)


    # if top_k:
    #     scores -= top_kth_iterative(scores, top_k)
    #     scores = tf.nn.relu(scores)
    #     weights_sum = tf.reduce_sum(scores, axis=-1, keep_dims=True)
    #     weights_sum = tf.maximum(weights_sum, 1e-6)  # Avoid division by 0.
    #     scores /= weights_sum

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.Print(output, ["output",  output[0]], summarize=1000)
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))

    if return_alphas:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        return output, scores

    return output


def top_kth_iterative(x, k):
    """Compute the k-th top element of x on the last axis iteratively.
    This assumes values in x are non-negative, rescale if needed.
    It is often faster than tf.nn.top_k for small k, especially if k < 30.
    Note: this does not support back-propagation, it stops gradients!
    Args:
      x: a Tensor of non-negative numbers of type float.
      k: a python integer.
    Returns:
      a float tensor of the same shape as x but with 1 on the last axis
      that contains the k-th largest number in x.
    """
    # The iterative computation is as follows:
    #
    # cur_x = x
    # for _ in range(k):
    #   top_x = maximum of elements of cur_x on the last axis
    #   cur_x = cur_x where cur_x < top_x and 0 everywhere else (top elements)
    #
    # We encode this computation in a TF graph using tf.foldl, so the inner
    # part of the above loop is called "next_x" and tf.foldl does the loop.
    def next_x(cur_x, _):
        top_x = tf.reduce_max(cur_x, axis=-1, keep_dims=True)
        return cur_x * tf.to_float(cur_x < top_x)
    # We only do k-1 steps of the loop and compute the final max separately.
    fin_x = tf.foldl(next_x, tf.range(k - 1), initializer=tf.stop_gradient(x),
                     parallel_iterations=2, back_prop=False)
    return tf.stop_gradient(tf.reduce_max(fin_x, axis=-1, keep_dims=True))


class VecAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


def calc_gauc(raw_arr, nick_index):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    last_index = 0
    gauc = 0.
    pv_sum = 0
    for idx in xrange(len(nick_index)):
        if nick_index[idx] != nick_index[last_index]:
            input_arr = raw_arr[last_index:idx]
            auc_val=calc_auc(input_arr)
            if auc_val >= 0.0:
                gauc += auc_val * len(input_arr)
                pv_sum += len(input_arr)
            else:
                pv_sum += len(input_arr) 
            last_index = idx
    return gauc / pv_sum
                

def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    if mask is not None:
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        if not forCnn:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output

