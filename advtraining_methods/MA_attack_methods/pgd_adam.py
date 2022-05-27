"""
Projected gradient decsent, adam solver
malware paper link: https://arxiv.org/abs/1812.08108
"""
from __future__ import print_function

import os
import sys
import warnings
import tqdm
import tensorflow as tf
import numpy as np
from tools import utils
from attacker.AttackerBase import AttackerBase
from tools.adam_optimizer import TensorAdam

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)


DEFAULT_PARAM = {
    'learning_rate': 0.01,
    'max_iteration': 1000,
    'force_iteration': False
}

class PGDAdam(AttackerBase):
    def __init__(self,
                 targeted_model,
                 **kwargs):
        super(PGDAdam, self).__init__()
        self.lr = DEFAULT_PARAM['learning_rate']
        self.model = targeted_model
        self.batch_size = 1
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.optimizer = TensorAdam(lr=self.lr)
        self.parse(**kwargs)
        self.model_inference()

    def parse(self,
              learning_rate=0.01,
              max_iteration=55,
              batch_size=50,
              force_iteration=False,
              **kwargs):
        self.lr = learning_rate
        self.iterations = max_iteration
        self.batch_size = 1
        self.optimizer = TensorAdam(lr=self.lr)
        self.force_iteration = force_iteration
        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def model_inference(self):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None, ])
        """
        作为一个还没被攻击就已经分类错误的样本是没有资格作为对抗样本的，所以要在攻击之前先看看这个样本是否分类正确。
        """
        self.output_pred = self.model.get_output_pred(self.x_input)
        self.ForceIteration = tf.placeholder(dtype=tf.bool)
        self.launch_an_attack = self.graph()


    def graph(self):
        init_state = self.optimizer.init_state([tf.zeros_like(self.x_input, dtype=tf.float32)])
        nest = tf.contrib.framework.nest
        scaled_max_extended = tf.maximum(  # broadcasting
            tf.multiply(self.scaled_clip_max,
                        tf.to_float(self.insertion_perm_array)) +  # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            # may be useful to reset the lower bound
            self.x_input  # upper bound for positions no perturbations allowed
        )
        scaled_min_extended = tf.minimum(  # broadcasting
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            self.x_input
        )

        def _cond(i, _1, _2, _attack_success):
            return tf.logical_and(
                tf.less(i, self.iterations),  # i < self.iterations 则为True,继续执行循环
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success))
                # ForceIteration为True，则忽略attack_success。ForceIteration为False，attack_success需要为True，
            )

        # 攻击成功

        def single_iteration(i, x_adv_tmp, flat_optim_state, attack_success_useless):
            def _loss_fn_wrapper(x_adv_tmp):  # 这里传logits，而不是直接传x，是因为后面我们的logits可能是混合logits
                output_logits = self.model.get_output_logits(x_adv_tmp)
                logits = output_logits
                return -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.y_input)

            curr_state = nest.pack_sequence_as(structure=init_state,
                                               flat_sequence=flat_optim_state)
            x_adv_tmp_list, new_optim_state = self.optimizer.minimize(_loss_fn_wrapper,
                                                                      [x_adv_tmp],
                                                                      curr_state)
            x_adv_tmp_clip = tf.clip_by_value(x_adv_tmp_list[0],
                                              clip_value_min=scaled_min_extended,
                                              clip_value_max=scaled_max_extended)
            x_adv_tmp_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_adv_tmp_clip)

            predict_x_adv_tmp_discrete = self.model.get_output_pred(x_adv_tmp_discrete)
            _attack_success = tf.logical_not(tf.equal(predict_x_adv_tmp_discrete, self.y_input))[0]  # 这里是self.batch_size=1的特化

            return i + 1, x_adv_tmp_clip, nest.flatten(new_optim_state), _attack_success

        flat_init_state = nest.flatten(init_state)
        iter_num, x_adv_batch, _, attack_success = tf.while_loop(_cond,
                                                                 single_iteration,
                                                                 (0, self.x_input, flat_init_state, False),
                                                                 maximum_iterations=self.iterations,
                                                                 back_prop=False)
        return iter_num, x_adv_batch, attack_success

    def perturb(self, attack_feature_vectors, attack_feature_labels, sess=None):
        # TF tensor
        # if self.is_init_graph is False:
        #    self.x_adv_batch = self.graph(self.model.x_input, self.model.y_input)

        try:
            input_data = utils.DataProducer(attack_feature_vectors,
                                            attack_feature_labels,
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
            raise IOError("PGD adam attack: Failed to load data and model parameters.")
        fool_num = 0
        iter_num_sum = 0
        x_adv = []
        x_ori = []
        x_adv_ori = []
        available_sample_num = 1e-6
        attack_success_record = np.zeros((attack_feature_vectors.shape[0]), dtype=bool)
        with sess.as_default():
            for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
                predict_clean_x_before_attack = sess.run(self.output_pred, feed_dict={self.x_input: x_input_var})
                if int(predict_clean_x_before_attack[0] != int(y_input_var[0])):
                    x_adv_ori.append(x_input_var[0])
                    continue
                available_sample_num += 1
                iter_num, x_adv_var, attack_success = sess.run(self.launch_an_attack,
                                                               feed_dict={self.x_input: x_input_var,
                                                                          self.y_input: y_input_var,
                                                                          self.ForceIteration: self.force_iteration})
                if bool(attack_success) is True: #注意这里一定要加一个强制类型转换，因为sess.run返回的并不是严格的bool，而是class<bool_>
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
        print("available_sample_num: ", available_sample_num)
        print("adv_accuracy: {:.2f}% ".format(adv_accuracy * 100))
        print("average iter_num: {:.2f}".format(iter_num_sum / available_sample_num))
        if x_adv.shape[0] != 0:
            perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv - x_ori) > 1e-6, axis=1))
            # perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv - x_ori), axis=1)))
            print("The average number of modified features is {:.2f}".format(perturbations_amount_l0))
        else:
            print("no sample is successfully attacked")

        return x_adv, x_ori, x_adv_ori, attack_success_record


"""
More specifically: The two namedtuples don't have the same sequence type. 
First structure type=list str=
                              [      <tf.Tensor 'while/Identity_1:0' shape=() dtype=int32>, 
                                     <tf.Tensor 'while/Identity_2:0' shape=(?, 10000) dtype=float32>, 
                                     [   <tf.Tensor 'while/Identity_3:0' shape=(?, 10000) dtype=float32>, 
                                         <tf.Tensor 'while/Identity_4:0' shape=() dtype=float32>, 
                                         <tf.Tensor 'while/Identity_5:0' shape=(?, 10000) dtype=float32>
                                     ], 
                                     <tf.Tensor 'while/Identity_6:0' shape=() dtype=bool>
                              ] has type list, 
while second structure type=tuple str=
                              (      <tf.Tensor 'while/add_6:0' shape=() dtype=int32>, 
                                     <tf.Tensor 'while/clip_by_value:0' shape=(?, 10000) dtype=float32>, 
                                     [   <tf.Tensor 'while/add_2:0' shape=(?, 10000) dtype=float32>, 
                                         <tf.Tensor 'while/add_1:0' shape=() dtype=float32>, 
                                         <tf.Tensor 'while/add_3:0' shape=(?, 10000) dtype=float32>
                                     ], 
                                     <tf.Tensor 'whilerided_slice:0' shape=() dtype=bool>
                              ) has type tuple
Entire first structure:
[., [., ., [., ., .], .]]
Entire second structure:
[., (., ., [., ., .], .)]
"""