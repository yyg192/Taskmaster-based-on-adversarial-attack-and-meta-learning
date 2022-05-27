import tensorflow as tf
from tools.DataProducer import DataProducer
import numpy as np
from tools.feature_reverser import DrebinFeatureReverse
import os
from timeit import default_timer
from tools.file_operation import read_pickle
from sklearn.model_selection import train_test_split
from tools import utils
from config import config
from config import modelSubstitute_Param
from config import model_Param_dict
import tqdm
from models.teacher_model import teacher_model
import random

"""
class DistillationMaterialProcess:
    def __init__(self,
                 feature_reverser,
                 input_dim,
                 target_model):
        self.feature_reverser = feature_reverser
        self.normalizer = feature_reverser.normalizer
        self.clip_min, self.clip_max = utils.get_min_max_bound(normalizer=self.normalizer)
        self.scaled_clip_min = utils.normalize_transform(np.reshape(self.clip_min, (1, -1)), normalizer=self.normalizer)
        self.scaled_clip_max = utils.normalize_transform(np.reshape(self.clip_max, (1, -1)), normalizer=self.normalizer)
        self.insertion_perm_array, self.removal_perm_array = feature_reverser.get_mod_array()
        self.iterations = 30
        self.force_iteration = False
        self.theta = 1.
        self.increase = bool(self.theta > 0)  # True
        self.model = target_model
        self.input_dim = input_dim
        self.output_dim = 2
        self.batch_size = 1
        self.ForceIteration = tf.placeholder(dtype=tf.bool)
        self.targeted_y_input = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.launch_an_attack = self.graph()

    def graph(self):
        _scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max,
                        self.insertion_perm_array) +  # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
            # may be useful to reset the lower bound
            self.model.x_input  # upper bound for positions no perturbations allowed
        )
        _scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, self.removal_perm_array) +
            tf.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
            self.model.x_input
        )
        if self.increase:
            search_domain = tf.reshape(tf.cast(self.model.x_input < _scaled_max_extended, tf.float32),
                                       [-1, self.input_dim])
        else:
            search_domain = tf.reshape(tf.cast(self.model.x_input > _scaled_min_extended, tf.float32),
                                       [-1, self.input_dim])

        y_in_init = tf.reshape(tf.one_hot(self.targeted_y_input, depth=self.output_dim), [-1, self.output_dim])

        def _cond(x_in, domain_in, i, _exist_modifiable_feature, _attack_success):
            return tf.logical_and(
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success)),
                tf.logical_and(tf.less(i, self.iterations), _exist_modifiable_feature)
            )
        def _body(x_in, domain_in, i, _useless1, _useless2):
            output_logits = self.model.get_output_logits(x_in)
            logits = output_logits
            preds = tf.nn.softmax(logits)
            # create the Jacobian graph
            list_derivatives = []
            for class_ind in range(self.output_dim):
                derivatives = tf.gradients(preds[:, class_ind], x_in)
                list_derivatives.append(derivatives[0])
            grads = tf.reshape(
                tf.stack(list_derivatives), shape=[self.output_dim, -1, self.input_dim])

            target_class = tf.reshape(
                tf.transpose(y_in_init, perm=[1, 0]), shape=[self.output_dim, -1, 1])
            other_classes = tf.cast(tf.not_equal(target_class, 1), tf.float32)

            grads_target = tf.reduce_sum(grads * target_class, axis=0)
            grads_other = tf.reduce_sum(grads * other_classes, axis=0)

            increase_coef = (4 * int(self.increase) - 2) \
                            * tf.cast(tf.equal(domain_in, 0), tf.float32)

            target_tmp = grads_target
            target_tmp -= increase_coef \
                          * tf.reduce_max(tf.abs(grads_target), axis=1, keepdims=True)
            target_sum = tf.reshape(target_tmp, shape=[-1, self.input_dim])

            other_tmp = grads_other
            other_tmp += increase_coef \
                         * tf.reduce_max(tf.abs(grads_other), axis=1, keepdims=True)
            other_sum = tf.reshape(other_tmp, shape=[-1, self.input_dim])

            # Create a mask to only keep features that match conditions
            if self.increase:
                scores_mask = ((target_sum > 0) & (other_sum < 0))
            else:
                scores_mask = ((target_sum < 0) & (other_sum > 0))

            # Extract the best malware feature
            scores = tf.cast(scores_mask, tf.float32) * (-target_sum * other_sum)
            best = tf.argmax(scores, axis=1)
            p1_one_hot = tf.one_hot(best, depth=self.input_dim)

            _exist_modifiable_features = (tf.reduce_sum(search_domain, axis=1) >= 1)
            _exist_modifiable_features_float = \
                tf.reshape(tf.cast(_exist_modifiable_features, tf.float32), shape=[-1, 1])  # shape: (?, 1)

            to_mod = p1_one_hot * _exist_modifiable_features_float
            domain_out = domain_in - to_mod

            to_mod_reshape = tf.reshape(
                to_mod, shape=([-1] + x_in.shape[1:].as_list()))

            if self.increase:
                x_out = tf.minimum(x_in + to_mod_reshape * self.theta,
                                   self.scaled_clip_max)
            else:
                x_out = tf.maximum(x_in + to_mod_reshape * self.theta,
                                   self.scaled_clip_min)

            # Increase the iterator, and check if all miss-classifications are done
            i_out = tf.add(i, 1)
            x_adv_tmp_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_out)
            predict_x_adv_tmp_discrete = self.model.get_output_pred(x_adv_tmp_discrete)
            _attack_success = tf.equal(predict_x_adv_tmp_discrete, self.targeted_y_input)[0]

            return x_out, domain_out, i_out, _exist_modifiable_features[0], _attack_success

        _x_adv_var, _2, _iter_num, _exist_modifiable_featues, attack_success = tf.while_loop(
            _cond,
            _body,
            [self.model.x_input, search_domain, 0, True, False]
        )

        return _x_adv_var, attack_success

    def generate_distillation_material(self,
                                       attack_feature_vectors,
                                       attack_feature_labels,
                                       sess):
        self.scaled_clip_min = utils.normalize_transform(np.reshape(self.clip_min, (1, -1)), normalizer=self.normalizer)
        self.scaled_clip_max = utils.normalize_transform(np.reshape(self.clip_max, (1, -1)), normalizer=self.normalizer)
        random_mask = random.sample(range(0, self.input_dim), int(self.input_dim / 2))
        self.scaled_clip_max[0][random_mask] = 0
        self.scaled_clip_min[0][random_mask] = 1

        target_labels = utils.get_other_classes_batch(self.output_dim, attack_feature_labels)
        input_data = utils.DataProducer(attack_feature_vectors, target_labels,
                                        batch_size=1, name='test')
        x_adv_all = []
        fool_num = 0
        available_sample_num = 1e-6
        for idx, x_input_var, tar_y_input_var in tqdm.tqdm(input_data.next_batch()):
            tar_label = tar_y_input_var[:, 0]
            predict_clean_x_before_attack = sess.run(self.model.y_pred_output,
                                                     feed_dict={self.model.x_input: x_input_var})
            if int(predict_clean_x_before_attack[0]) == int(tar_y_input_var[0][0]):
                x_adv_all.append(x_input_var[0])
                available_sample_num += 1
                continue

            available_sample_num += 1
            x_adv_var, attack_success = sess.run(self.launch_an_attack,
                                                 feed_dict={self.model.x_input: x_input_var,
                                                            self.targeted_y_input: tar_label,
                                                            self.ForceIteration: self.force_iteration})
            x_adv_discrete_var = utils.map_to_discrete_domain(self.normalizer, x_adv_var)
            x_adv_all.append(x_adv_discrete_var[0])
            if bool(attack_success) is True:
                fool_num += 1

        x_adv_all = np.array(x_adv_all)
        adv_accuracy = 1 - fool_num / available_sample_num
        print("adv_accuracy: {:.2f}% ".format(adv_accuracy * 100))

        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_all - attack_feature_vectors) > 1e-6, axis=1))
        print("The number of modified features is {:.2f}".format(perturbations_amount_l0))
        return np.array(x_adv_all)

"""


class DistillationMaterialProcess:
    def __init__(self,
                 feature_reverser,
                 input_dim,
                 target_model):
        self.feature_reverser = feature_reverser
        self.normalizer = feature_reverser.normalizer
        self.clip_min, self.clip_max = utils.get_min_max_bound(normalizer=self.normalizer)
        self.scaled_clip_min = utils.normalize_transform(np.reshape(self.clip_min, (1, -1)), normalizer=self.normalizer)
        self.scaled_clip_max = utils.normalize_transform(np.reshape(self.clip_max, (1, -1)), normalizer=self.normalizer)
        self.insertion_perm_array, self.removal_perm_array = feature_reverser.get_mod_array()
        self.iterations = 30
        self.force_iteration = False
        self.model = target_model
        self.input_dim = input_dim
        self.output_dim = 2
        self.batch_size = 1
        self.ord = "l2"
        self.epsilon = 100.
        self.launch_an_attack = self.graph()

    def graph(self):
        scaled_max_extended = tf.maximum(  # broadcasting
            tf.multiply(self.scaled_clip_max,
                        tf.to_float(self.insertion_perm_array)) +  # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            # may be useful to reset the lower bound
            self.model.x_input  # upper bound for positions no perturbations allowed
        )
        scaled_min_extended = tf.minimum(  # broadcasting
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            self.model.x_input
        )
        output_logits = self.model.get_output_logits(self.model.x_input)
        output_proba = tf.nn.softmax(output_logits)
        output_pred = tf.reduce_max(output_proba, axis=1, keepdims=True)
        # 这个preds的shape是 [batchsize,1] 严格来说也是二维的。只不过第二个维度的元素只有一个，而y_proba则是[batchsize,2]
        # preds = tf.reduce_max(self.model.y_proba, axis=1, keepdims=True)  # 它这里keepdims=True了 不然它就是[batchsize]了
        y = tf.to_float(tf.equal(output_pred, output_proba))  # 现在这里的y就是one-hot的0和1的向量了，
        y = tf.stop_gradient(y)  # 感觉这个没有必要啊
        y = y / tf.reduce_sum(y, axis=1, keepdims=True)  # 这句话感觉没有存在的必要

        # label_masking = tf.one_hot(self.model.y_input, 2, on_value=1., off_value=0., dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits)
        # gradient
        grad, = tf.gradients(loss, self.model.x_input)

        if self.ord == 'l-infinity':
            perturbations = utils.optimize_linear(grad, eps=self.epsilon)
        elif self.ord == 'l1':
            perturbations = utils.optimize_linear(grad, eps=self.epsilon, ord=1)
        elif self.ord == 'l2':
            perturbations = utils.optimize_linear(grad, eps=self.epsilon, ord=2)
        else:
            raise ValueError("Only 'l1', 'l2', 'l-infinity' are supported.")
        x_adv_tmp = self.model.x_input + perturbations
        x_adv_tmp_clip = tf.clip_by_value(x_adv_tmp,
                                          clip_value_min=scaled_min_extended,
                                          clip_value_max=scaled_max_extended)
        # x_adv_tmp_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_adv_tmp_clip)
        # predict_x_adv_tmp_discrete = self.model.get_output_pred(x_adv_tmp_discrete)
        # _attack_success = tf.logical_not(tf.equal(predict_x_adv_tmp_discrete, self.model.y_input))[0]
        return x_adv_tmp_clip  # , _attack_success

    def generate_distillation_material(self,
                                       attack_feature_vectors,
                                       attack_feature_labels,
                                       sess):
        """
        self.scaled_clip_min = utils.normalize_transform(np.reshape(self.clip_min, (1, -1)), normalizer=self.normalizer)
        self.scaled_clip_max = utils.normalize_transform(np.reshape(self.clip_max, (1, -1)), normalizer=self.normalizer)
        random_mask = random.sample(range(0, self.input_dim), int(self.input_dim / 2))
        self.scaled_clip_max[0][random_mask] = 0
        self.scaled_clip_min[0][random_mask] = 1
        """
        input_data = utils.DataProducer(attack_feature_vectors, attack_feature_labels,
                                        batch_size=1, name='test')
        x_adv_all = []
        available_sample_num = 1e-6
        for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
            available_sample_num += 1
            x_adv_var = sess.run(self.launch_an_attack,
                                 feed_dict={self.model.x_input: x_input_var,
                                            self.model.y_input: y_input_var})
            x_adv_all.append(x_adv_var[0])

        x_adv_all = np.array(x_adv_all)
        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_all - attack_feature_vectors) > 1e-6, axis=1))
        print("The number of modified features is {:.2f}".format(perturbations_amount_l0))
        return np.array(x_adv_all)


def BASIC_DNN_GRAPH(
        x_input,
        name,
        hidden_neurons,
        output_dim=2,
        trainable=True,
        reuse=False):
    if len(hidden_neurons) == 1:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense_in, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 2:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense1, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 3:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            """
            backbone
            """
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[2], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE2_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense2, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 4:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            """
            backbone
            """
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[2], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE2_BACKBONE")
            dense3 = tf.layers.dense(inputs=dense2, units=hidden_neurons[3], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE3_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense3, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 5:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            """
            backbone
            """
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[2], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE2_BACKBONE")
            dense3 = tf.layers.dense(inputs=dense2, units=hidden_neurons[3], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE3_BACKBONE")
            dense4 = tf.layers.dense(inputs=dense3, units=hidden_neurons[4], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE4_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense4, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    else:
        raise ValueError("The layer of teacher model must less equal than 5 layers")
    return dense_out


class SubstituteModel:
    def __init__(self,
                 hyper_parameter,
                 is_trainable=False,
                 reuse=False,
                 ):
        """
        变量接收区
        """
        self.hp_params = utils.ParamWrapper(hyper_parameter)
        self.is_trainable = is_trainable
        self.distillation_iterations = 3
        """
        执行区
        """
        self.nn = BASIC_DNN_GRAPH  ##### 注意，我们可以通过修改graph函数来更改我们的模型 #####
        self.model_graph(reuse=reuse)

    def model_graph(self, reuse=False):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.hp_params.input_dim], name='X')
        self.y_input_logits = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Y_LOGITS')
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='Y')
        self.is_backbone_training = tf.placeholder(dtype=tf.bool, name="BACKBONE_TRAINING")  # 这个用于dropout和batchnorm的
        tf.set_random_seed(self.hp_params.random_seed)
        self.output_logits = self.nn(self.x_input,
                                     name=self.hp_params.model_name,
                                     hidden_neurons=self.hp_params.hidden_neurons,
                                     output_dim=self.hp_params.output_dim,
                                     trainable=self.is_trainable,
                                     reuse=reuse)

        # self.distillation_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input_logits,
        #                                                                    logits=self.output_logits)
        self.distillation_entropy = tf.losses.mean_squared_error(self.y_input_logits,
                                                                 self.output_logits)

        self.output_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_input,
            logits=self.output_logits
        )

        self.output_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input,
            logits=self.output_logits
        )

        self.y_pred_output = tf.argmax(self.output_logits, axis=1, output_type=tf.int32)

        self.accuracy_output = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred_output, self.y_input))
        )

        self.global_train_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer_1 = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.output_cross_entropy,
                                                                                             global_step=self.global_train_step)
            self.optimizer_2 = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.distillation_entropy,
                                                                                             global_step=self.global_train_step)

    def get_output_logits(self, x_tensor):
        fc_output_logits = self.nn(x_input=x_tensor,
                                   name=self.hp_params.model_name,
                                   hidden_neurons=self.hp_params.hidden_neurons,
                                   output_dim=self.hp_params.output_dim,
                                   trainable=False,
                                   reuse=True)
        return fc_output_logits

    def get_output_pred(self, x_tensor):
        output_logits = self.get_output_logits(x_tensor)
        y_pred_output = tf.argmax(output_logits, axis=1, output_type=tf.int32)
        return y_pred_output

    def load_param(self, load_dir, sess):
        cur_checkpoint = tf.train.latest_checkpoint(load_dir)
        meta_dir = cur_checkpoint + ".meta"
        restorer = tf.train.import_meta_graph(meta_dir)
        restorer.restore(sess, cur_checkpoint)
        # print("succssfully resore parmater from: {}".format(cur_checkpoint))
        return restorer

    def distillation(self, X, Y,
                     sess,
                     substitute_saver=None,
                     save_checkpoint_dir=None):
        train_x, valid_x, train_logits_labels, valid_logits_labels = \
            train_test_split(X, Y, test_size=0.1, random_state=30)
        train_input = DataProducer(train_x, train_logits_labels, self.hp_params.batch_size,
                                   n_epochs=self.hp_params.distillation_n_epochs)
        valid_input = DataProducer(valid_x, valid_logits_labels, self.hp_params.batch_size,
                                   n_epochs=self.hp_params.distillation_n_epochs)

        sess.run(tf.global_variables_initializer())
        """
        saver = None
        if load_checkpoint_dir is not None:
            saver = self.load_param(load_checkpoint_dir, sess)
        """

        training_time = 0.0
        train_input.reset_cursor()
        output_steps = 100
        for step_idx, X_train_batch, Y_train_batch in train_input.next_batch():
            train_dict = {
                self.x_input: X_train_batch,
                self.y_input_logits: Y_train_batch
            }
            if (step_idx + 1) % output_steps == 0 or step_idx+1 ==train_input.mini_batches:
                print('Step {}/{}'.format(step_idx + 1, train_input.steps))
                valid_input.reset_cursor()
                valid_loss_list = [sess.run(self.distillation_entropy,
                                            feed_dict={self.x_input: X_valid_batch,
                                                       self.y_input_logits: Y_valid_batch})
                                   for [_, X_valid_batch, Y_valid_batch] in valid_input.next_batch()
                                   ]
                valid_loss_list = np.array(valid_loss_list)
                average_valid_loss = np.mean(valid_loss_list)
                print("average_valid_loss: {:.5f}".format(average_valid_loss))

            start = default_timer()
            sess.run(self.optimizer_2, feed_dict=train_dict)
            end = default_timer()
            training_time = training_time + end - start

        if save_checkpoint_dir is None and substitute_saver is not None:
            raise ValueError("The save_checkpoint_dir is None but substitute_saver is not None")
        elif save_checkpoint_dir is not None and substitute_saver is None:
            raise ValueError("The save checkpoint_dir is not None but substitute saver is None")
        elif save_checkpoint_dir is not None and substitute_saver is not None:
            if os.path.exists(save_checkpoint_dir) is None:
                os.makedirs(save_checkpoint_dir)
                substitute_saver.save(sess, os.path.join(save_checkpoint_dir, 'checkpoint'),
                                      global_step=self.global_train_step)
            print("save backbone model")
        else:
            pass

    def train_backbone(self, X, Y,
                       sess,
                       save_checkpoint_dir=None):
        n_epochs = self.hp_params.warmup_n_epochs
        train_x, valid_x, train_y, valid_y = \
            train_test_split(X, Y, test_size=0.1, random_state=30)
        train_input = DataProducer(train_x, train_y, self.hp_params.batch_size, n_epochs=n_epochs)
        valid_input = DataProducer(valid_x, valid_y, self.hp_params.batch_size, n_epochs=n_epochs)
        # global_train_step = tf.train.get_or_create_global_step()
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #    optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.output_cross_entropy,
        #                                                                              global_step=global_train_step)

        sess.run(tf.global_variables_initializer())
        """
        saver = None
        if load_checkpoint_dir is not None:
            saver = self.load_param(load_checkpoint_dir, sess)
        """
        training_time = 0.0
        train_input.reset_cursor()
        output_steps = 50  ##############################################################################
        for step_idx, X_train_batch, Y_train_batch in train_input.next_batch():
            train_dict = {
                self.x_input: X_train_batch,
                self.y_input: Y_train_batch
            }
            if (step_idx + 1) % output_steps == 0:
                print('Step {}/{}'.format(step_idx + 1, train_input.steps))
                valid_input.reset_cursor()
                valid_result_list = [sess.run([self.accuracy_output, self.y_pred_output, self.output_cross_entropy],
                                              feed_dict={self.x_input: X_valid_batch,
                                                         self.y_input: Y_valid_batch
                                                         })
                                     for [_, X_valid_batch, Y_valid_batch] in valid_input.next_batch()
                                     ]
                valid_result = np.array(valid_result_list)
                _acc = np.mean(valid_result[:, 0])
                _pred_y = np.concatenate(valid_result[:, 1])
                average_loss = np.mean(valid_result[:, 2])

                from sklearn.metrics import f1_score
                _f1_score = f1_score(valid_y, _pred_y[:valid_y.shape[0]])
                print('    validation accuracy {:.5}%'.format(_acc * 100))
                print('    validation f1 score {:.5}%'.format(_f1_score * 100))
                print('    average_loss {:.5}'.format(average_loss))
            start = default_timer()
            sess.run(self.optimizer_1, feed_dict=train_dict)
            end = default_timer()
            training_time = training_time + end - start
        if save_checkpoint_dir is not None:
            if os.path.exists(save_checkpoint_dir) is None:
                os.makedirs(save_checkpoint_dir)
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(save_checkpoint_dir, 'checkpoint'), global_step=self.global_train_step)
            print("save backbone model")
        else:
            print("without saving the backbone model")


TEACHERS = ['model_A', 'model_B', 'model_C', 'model_D', 'model_E', 'model_F', 'model_G']
MODEL_NAME = "model_substitute"


def stealing_feature():
    global DistillationMaterialProcess
    warmup_checkpoint = "/home/newdisk_1/yugang_yang/malware_evasion_attack/adv_training/model_substitute/warmup_checkpoint"
    ori_malware_feature_vectors = read_pickle(config.get('feature.drebin', 'malware_sample_vectors_1024'))
    ori_malware_feature_labels = read_pickle(config.get('feature.drebin', 'malware_sample_labels_1024'))
    ori_benware_feature_vectors = read_pickle(config.get('feature.drebin', 'benware_sample_vectors_1024'))
    ori_benware_feature_labels = read_pickle(config.get('feature.drebin', 'benware_sample_labels_1024'))
    feature_reverser = DrebinFeatureReverse(feature_mp='binary')
    insertion_perm_array, removal_perm_array = feature_reverser.get_mod_array()
    all_feature_vectors = np.concatenate((ori_malware_feature_vectors, ori_benware_feature_vectors), axis=0)
    all_feature_labels = np.concatenate((ori_malware_feature_labels, ori_benware_feature_labels), axis=0)

    """
    substitute model define and warm up training
    """
    substitute_graph = tf.Graph()
    sess_substitute = tf.Session(graph=substitute_graph)
    with substitute_graph.as_default():
        substitute_model = SubstituteModel(hyper_parameter=modelSubstitute_Param, is_trainable=True)
        DMP = DistillationMaterialProcess(feature_reverser=feature_reverser,
                                          input_dim=1024,
                                          target_model=substitute_model)
        substitute_model.train_backbone(all_feature_vectors, all_feature_labels,
                                        sess=sess_substitute,
                                        save_checkpoint_dir=warmup_checkpoint,
                                        load_checkpoint_dir=None)

    """
    teacher model define
    """
    teacher_models_graph = []
    teacher_models_sess = []
    teacher_models = []
    for j, teacher_name in enumerate(TEACHERS):
        graph_j = tf.Graph()
        sess_j = tf.Session(graph=graph_j)
        with graph_j.as_default():
            t = teacher_model(hyper_parameter=model_Param_dict[teacher_name],
                              is_trainable=False)
            sess_j.run(tf.global_variables_initializer())
            teacher_models.append(t)
        teacher_models_graph.append(graph_j)
        teacher_models_sess.append(sess_j)
    last_checkpoint = warmup_checkpoint
    for i in range(30):
        """
        teacher model load parameter
        """
        for j, teacher_name in enumerate(TEACHERS):
            graph_j = teacher_models_graph[j]
            sess_j = teacher_models_sess[j]
            with graph_j.as_default():
                with sess_j.as_default():
                    teacher = teacher_models[j]
                    teacher_model_load_dir = config.get('advtraining.' + teacher_name,
                                                        'model_checkpoint') + "/adv" + str(i)
                    teacher.load_param(teacher_model_load_dir, sess_j)
        """
        substitute model load parameter
        """
        with substitute_graph.as_default():
            with sess_substitute.as_default():
                substitute_saver = substitute_model.load_param(last_checkpoint, sess_substitute)
        substitute_model_save_dir = config.get('advtraining.substitute_model', 'model_checkpoint') + "/adv" + str(i)
        last_checkpoint = substitute_model_save_dir
        for cur_iter in range(substitute_model.distillation_iterations):
            print(" i = {}  cur_ier = {}".format(i, cur_iter))
            random_num = np.random.randint(0, high=999)
            part_feature_vectors, _, part_feature_labels, _ = \
                train_test_split(all_feature_vectors, all_feature_labels, test_size=0.7, random_state=random_num)

            """
            with substitute_graph.as_default():
                with sess_substitute.as_default():
                    if cur_iter == 0:
                        substitute_saver = substitute_model.load_param(last_checkpoint, sess_substitute)
                    print("substitute_model.accuracy_output:",
                          sess_substitute.run(substitute_model.accuracy_output,
                                              feed_dict={substitute_model.x_input: part_feature_vectors,
                                                         substitute_model.y_input: part_feature_labels}))
            """

            #teacher_output_logits_total = []

            j = random.randint(0, len(TEACHERS))
            graph_j = teacher_models_graph[j]
            sess_j = teacher_models_sess[j]
            with graph_j.as_default():
                with sess_j.as_default():
                    teacher = teacher_models[j]
                    distillation_material = \
                        DMP.generate_distillation_material(part_feature_vectors,
                                                           part_feature_labels,
                                                           sess_substitute)

                    teacher_output_logit = sess_j.run(teacher.softmax_output_logits,
                                                      feed_dict={teacher.x_input: distillation_material})

                    #teacher_output_logits_total.append(teacher_output_logit)

            with substitute_graph.as_default():
                with sess_substitute.as_default():
                    if cur_iter+1 == substitute_model.distillation_iterations:
                        substitute_model.distillation(distillation_material, teacher_output_logit,
                                                      sess=sess_substitute,
                                                      save_checkpoint_dir=substitute_model_save_dir,
                                                      substitute_saver=substitute_saver)
                    else:
                        substitute_model.distillation(distillation_material, teacher_output_logit,
                                                      sess=sess_substitute,
                                                      save_checkpoint_dir=None,
                                                      substitute_saver=None)

            # teacher_output_logits_total = np.array(teacher_output_logits_total)

            # teacher_output_logits_mean = np.mean(teacher_output_logits_total, axis=0)
        substitute_adv_samples_save_dir = config.get('advtraining.substitute_model', 'adv_samples') + "/adv" + str(i)


if __name__ == "__main__":
    stealing_feature()
