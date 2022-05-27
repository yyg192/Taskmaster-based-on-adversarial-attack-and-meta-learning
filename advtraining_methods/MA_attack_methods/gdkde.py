"""GD-KED:https://arxiv.org/abs/1708.06131"""

import os
import sys
import warnings
from tools import utils
import tqdm
import numpy as np
from attacker.AttackerBase import AttackerBase
import tensorflow as tf

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)

DEFAULT_PARAM = {
    'step_size': 1.,
    'max_iteration': 1000,
    'negative_data_num': 100,
    'kernel_width': 10.,
    'lambda_factor': 100.,
    'distance_max': 20.,  # indicator of projection during the iteration
    'xi': 1e-6,  # terminate the iteration
    'batch_size': 1,
    'force_iteration': False
}


class GDKDE(AttackerBase):
    def __init__(self,
                 targeted_model,
                 attack_feature_vectors,
                 attack_feature_labels,
                 **kwargs):
        super(GDKDE, self).__init__()
        self.model=targeted_model
        self.step_size = DEFAULT_PARAM['step_size']
        self.neg_data_num = DEFAULT_PARAM['negative_data_num']
        self.kernel_width = DEFAULT_PARAM['kernel_width']
        self.lambda_factor = DEFAULT_PARAM['lambda_factor']
        self.d_max = DEFAULT_PARAM['distance_max']
        self.xi = DEFAULT_PARAM['xi']
        self.attack_feature_vectors = attack_feature_vectors
        self.attack_feature_labels = attack_feature_labels
        self.batch_size = 1
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.parse(**kwargs)
        self.neg_dataX = self._load_neg_data(self.neg_data_num)
        # TF tensor
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.output_pred = self.model.get_output_pred(self.x_input)
        self.ForceIteration = tf.placeholder(dtype=tf.bool)
        self.launch_an_attack = self.graph()

    def parse(self, step_size=1, max_iteration=50, negative_data_num=100,
              kernel_width=10., lambda_factor=100, distance_max=20., xi=1e-6,
              batch_size=50, force_iteration=False, **kwargs):
        self.step_size = step_size
        self.iterations = max_iteration
        self.neg_data_num = negative_data_num
        self.kernel_width = kernel_width
        self.lambda_factor = lambda_factor
        self.d_max = distance_max
        self.xi = xi
        self.batch_size = 1
        self.force_iteration = force_iteration

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def _load_neg_data(self, max_num=100):
        if not isinstance(max_num, int) and max_num < 0:
            raise TypeError("Input must be an positive interger.")

        negative_data_idx = (self.attack_feature_labels == 1)
        neg_dataX = self.attack_feature_vectors[negative_data_idx]
        if len(neg_dataX) == 0:
            raise ValueError("No negative data.")
        elif len(neg_dataX) < max_num:
            np.random.seed(0)
            return neg_dataX[np.random.choice(len(neg_dataX), max_num, replace=True)]
        else:
            np.random.seed(0)
            return neg_dataX[np.random.choice(len(neg_dataX), max_num, replace=False)]

    @staticmethod
    def _laplician_kernel(x1, x2, w):
        '''
        calculate the laplician kernel with x1 and x2
        :param x1: input one, shape is [batch_size, input_dim]
        :param x2: input two, shape is [number_of_negative_data, input_dim]
        :param w: the width of kernal
        :return: the laplician kernel value, shape is [batch_size, number_of_negative_data]
        '''
        return tf.exp(-1 * tf.reduce_sum(tf.abs(tf.expand_dims(x1, 1) - tf.expand_dims(x2, 0)),
                                         axis=2) / w)

    def graph(self):
        negX = tf.constant(self.neg_dataX, dtype=tf.float32)
        scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) +
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            self.x_input
        )
        scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            self.x_input
        )

        def _cond(i, _, _attack_success):
            return tf.logical_and(
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success)),
                tf.less(i, self.iterations)
            )

        def _body(i, x_in, useless):
            def F(_x_in):
                """在这里修改"""
                output_logits = self.model.get_output_logits(_x_in)

                logits = output_logits
                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=self.y_input)
                kernel = self._laplician_kernel(_x_in, negX, self.kernel_width)
                kde = tf.divide(self.lambda_factor, self.neg_data_num) * tf.reduce_sum(kernel, -1)
                return tf.reduce_mean(y_xent - kde)

            loss1 = F(x_in)
            grad = tf.gradients(loss1, x_in)[0]
            perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size), ord=2)
            x_out = x_in - perturbations
            x_out = tf.clip_by_value(x_out,
                                     clip_value_min=scaled_min_extended,
                                     clip_value_max=scaled_max_extended)
            i_out = tf.add(i, 1)
            x_adv_tmp_discrete = utils.map_to_discrete_domain_TF(self.normalizer,
                                                                 x_out)
            predict_x_adv_tmp_discrete = self.model.get_output_pred(x_adv_tmp_discrete)
            _attack_success = tf.equal(predict_x_adv_tmp_discrete, self.y_input)[0]
            return i_out, x_out, _attack_success

        _iter_num, _x_adv_var, attack_success = tf.while_loop(_cond,
                                                              _body,
                                                              [0, self.x_input, False])

        return _iter_num, _x_adv_var, attack_success

    def perturb(self, attack_feature_vectors, attack_feature_labels, sess=None)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               :
        self.target_labels = utils.get_other_classes_batch(self.output_dim, attack_feature_labels)
        try:
            input_data = utils.DataProducer(attack_feature_vectors,
                                            self.target_labels,
                                            batch_size=self.batch_size,
                                            name='test')
            # load model parameters
            sess_close_flag = False
            if sess is None:
                cur_checkpoint = tf.train.latest_checkpoint(self.model.final_checkpoint_dir)
                config_gpu = tf.ConfigProto(log_device_placement=False)
                config_gpu.gpu_options.allow_growth = True
                sess = tf.Session(config=config_gpu)
                saver = tf.train.Saver()
                saver.restore(sess, cur_checkpoint)
                sess_close_flag = True
        except IOError as ex:
            raise IOError("Failed to load data and model parameters.")

        x_adv = []
        x_ori = []
        x_adv_ori = []
        fool_num = 0
        iter_num_sum = 0
        available_sample_num = 1e-6
        attack_success_record = np.zeros((attack_feature_vectors.shape[0]), dtype=bool)
        with sess.as_default():
            for idx, x_input_var, tar_y_input_var in tqdm.tqdm(input_data.next_batch()):
                # boundary
                _batch_x_adv_container = []
                tar_label = tar_y_input_var[:, 0]
                predict_clean_x_before_attack = sess.run(self.output_pred,
                                                         feed_dict={self.x_input: x_input_var})
                if int(predict_clean_x_before_attack[0]) == int(tar_label[0]):
                    x_adv_ori.append(x_input_var[0])
                    continue
                available_sample_num += 1
                iter_num, x_adv_var, attack_success = \
                    sess.run(self.launch_an_attack,
                             feed_dict={self.x_input: x_input_var,
                                        self.y_input: tar_label,
                                        self.ForceIteration: self.force_iteration})
                if bool(attack_success) is True:
                    x_adv_discrete_var = utils.map_to_discrete_domain(self.normalizer, x_adv_var)
                    x_adv.append(x_adv_discrete_var[0])
                    x_ori.append(x_input_var[0])
                    fool_num += 1
                    iter_num_sum += iter_num
                    attack_success_record[idx] = True
                    x_adv_ori.append(x_adv_discrete_var[0])
                else:
                    x_adv_ori.append(x_input_var[0])
            if sess_close_flag:
                sess.close()

        x_adv = np.array(x_adv)
        x_ori = np.array(x_ori)
        x_adv_ori = np.array(x_adv_ori)
        adv_accuracy = 1 - fool_num / available_sample_num
        print("adv_accuracy: {:.2f}% ".format(adv_accuracy * 100))
        print("average iter_num: {:.2f}".format(iter_num_sum / available_sample_num))
        if x_adv.shape[0] != 0:
            perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv - x_ori) > 1e-6, axis=1))
            print("The number of modified features is {:.2f}".format(perturbations_amount_l0))
        else:
            print("No sample is successfully attacked")

        return x_adv, x_ori, x_adv_ori, attack_success_record