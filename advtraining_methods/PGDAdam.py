import tqdm
import tensorflow as tf
from attacker.AttackerBase import AttackerBase
from tools.adam_optimizer import TensorAdam
from tools import utils
import numpy as np
import random
DEFAULT_PARAM = {
    'learning_rate': 0.01,
    'max_iteration': 1000,
    'force_iteration': False
}

class PGDAdam(AttackerBase):
    def __init__(self,
                 targeted_model,
                 maximum_iterations,
                 force_iteration,
                 random_mask,
                 mask_rate,
                 ):
        super(PGDAdam, self).__init__()
        self.lr = DEFAULT_PARAM['learning_rate']
        self.batch_size = 1
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.maximum_iterations = maximum_iterations
        self.random_mask = random_mask
        self.force_iteration = force_iteration
        self.mask_rate = mask_rate
        self.model = targeted_model
        self.ForceIteration = tf.placeholder(dtype=tf.bool)
        self.random_mask_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.optimizer = TensorAdam(lr=self.lr)
        self.launch_an_attack_teacher = self.graph_teacher()
        self.launch_an_attack_substitute = self.graph_substitute()

    def graph_teacher(self):
        init_state = self.optimizer.init_state([tf.zeros_like(self.model.x_input, dtype=tf.float32)])
        nest = tf.contrib.framework.nest
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

        def _cond(i, _1, _2, _attack_success):
            return tf.logical_and(
                tf.less(i, self.iterations),  # i < self.iterations 则为True,继续执行循环
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success))
                # ForceIteration为True，则忽略attack_success。ForceIteration为False，attack_success需要为True，
            )

        def single_iteration(i, x_adv_tmp, flat_optim_state, attack_success_useless):
            def _loss_fn_wrapper(x_adv_tmp):  # 这里传logits，而不是直接传x，是因为后面我们的logits可能是混合logits
                output_logits = self.model.get_output_logits(x_adv_tmp)
                logits = output_logits
                return -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.model.y_input)

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
            _attack_success = tf.logical_not(tf.equal(predict_x_adv_tmp_discrete, self.model.y_input))[
                0]  # 这里是self.batch_size=1的特化

            return i + 1, x_adv_tmp_clip, nest.flatten(new_optim_state), _attack_success

        flat_init_state = nest.flatten(init_state)
        iter_num, x_adv_batch, _, attack_success = tf.while_loop(_cond,
                                                                 single_iteration,
                                                                 (0, self.model.x_input, flat_init_state, False),
                                                                 maximum_iterations=self.iterations,
                                                                 back_prop=False)
        return iter_num, x_adv_batch, attack_success

    def graph_substitute(self):
        init_state = self.optimizer.init_state([tf.zeros_like(self.model.x_input, dtype=tf.float32)])
        nest = tf.contrib.framework.nest
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

        def _cond(i, _1, _2, _attack_success):
            return tf.logical_and(
                tf.less(i, self.iterations),  # i < self.iterations 则为True,继续执行循环
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success))
                # ForceIteration为True，则忽略attack_success。ForceIteration为False，attack_success需要为True，
            )

        def single_iteration(i, x_adv_tmp, flat_optim_state, attack_success_useless):
            def _loss_fn_wrapper(x_adv_tmp):  # 这里传logits，而不是直接传x，是因为后面我们的logits可能是混合logits
                output_logits = self.model.get_output_logits(x_adv_tmp)
                logits = output_logits
                return -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.model.y_input)

            curr_state = nest.pack_sequence_as(structure=init_state,
                                               flat_sequence=flat_optim_state)
            x_adv_tmp_list, new_optim_state = self.optimizer.minimize(_loss_fn_wrapper,
                                                                      [x_adv_tmp],
                                                                      curr_state)
            max_random_mask_T = self.random_mask_TS
            min_random_mask_T = tf.multiply(tf.subtract(self.random_mask_TS, 1), -1)
            mask_scaled_max_extended = tf.multiply(scaled_max_extended, max_random_mask_T)
            mask_scaled_min_extended = tf.clip_by_value(scaled_min_extended + min_random_mask_T,
                                                        clip_value_min=self.scaled_clip_min,
                                                        clip_value_max=self.scaled_clip_max)

            x_adv_tmp_clip = tf.clip_by_value(x_adv_tmp_list[0],
                                              clip_value_min=mask_scaled_min_extended,
                                              clip_value_max=mask_scaled_max_extended)
            predict_x_adv_tmp = self.model.get_output_pred(x_adv_tmp_clip)
            _attack_success = tf.logical_not(tf.equal(predict_x_adv_tmp, self.model.y_input))[
                0]  # 这里是self.batch_size=1的特化

            return i + 1, x_adv_tmp_clip, nest.flatten(new_optim_state), _attack_success

        flat_init_state = nest.flatten(init_state)
        iter_num, x_adv_batch, _, attack_success = tf.while_loop(_cond,
                                                                 single_iteration,
                                                                 (0, self.model.x_input, flat_init_state, False),
                                                                 maximum_iterations=self.iterations,
                                                                 back_prop=False)
        return iter_num, x_adv_batch, attack_success

    def generate_attack_samples_teacher(self,
                                        attack_feature_vectors,
                                        attack_feature_labels,
                                        samples_save_dir,
                                        sess,
                                        flag="malware"):
        """
        teacher专用
        """
        input_data = utils.DataProducer(attack_feature_vectors, attack_feature_labels,
                                        batch_size=1, name='test')  #### batch_size 只能为1
        x_adv_success = []
        x_ori = []
        x_adv_all = []
        available_sample_num = 1e-6
        fool_num = 1e-6
        iter_num_sum = 0
        for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
            predict_x_input = sess.run(self.model.y_pred_output,
                                       feed_dict={self.model.x_input: x_input_var})
            if int(predict_x_input[0]) != int(np.argmax(y_input_var, axis=-1)[0]):
                x_adv_all.append(x_input_var[0])
                continue

            available_sample_num += 1
            iter_num, x_adv_var, _attack_success = sess.run(self.launch_an_attack_teacher,
                                                            feed_dict={self.model.x_input: x_input_var,
                                                                       self.model.y_input: y_input_var,
                                                                       self.ForceIteration: self.force_iteration})
            x_adv_all.append(x_adv_var[0])
            if bool(_attack_success) is True:
                x_adv_success.append(x_adv_var[0])
                x_ori.append(x_input_var[0])
                fool_num += 1
                iter_num_sum += iter_num

        x_ori = np.array(x_ori)
        x_adv_success = np.array(x_adv_success)
        x_adv_all = np.array(x_adv_all)
        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_success - x_ori) > 1e-6, axis=1))
        print("average modified samples number is :{:.2f} "
              "attack_success_rate is {:.2f}".format(perturbations_amount_l0,
                                                     fool_num / available_sample_num * 100))
        if samples_save_dir is not None:
            utils.dump_pickle(x_adv_success, samples_save_dir + "/x_adv_success_"+flag)
            utils.dump_pickle(x_adv_all, samples_save_dir + "/x_adv_all_"+flag)
            utils.dump_pickle(x_ori, samples_save_dir + "/x_ori_"+flag)
        return x_adv_success, x_adv_all

    def generate_attack_samples_substitute(self,
                                           attack_feature_vectors,
                                           attack_feature_labels,
                                           require_sample_nums,
                                           sess):
        """
        simulator专用！！
        """
        if len(attack_feature_vectors.shape) == 1:
            attack_feature_vectors = np.array([attack_feature_vectors])
            attack_feature_labels = np.array([attack_feature_labels])
        elif len(attack_feature_vectors.shape) == 2:
            assert (len(attack_feature_vectors) == 1)
        else:
            raise ValueError("Only one sample can be received")
        input_data = utils.DataProducer(attack_feature_vectors, attack_feature_labels,
                                        batch_size=1, name='test')  #### batch_size 只能为1
        x_adv_all = []
        x_ori = []
        logits = []
        available_sample_num = 1e-6
        fool_num = 1e-6
        iter_num_sum = 0
        for idx, x_input_var, y_input_var in input_data.next_batch():
            for _ in range(require_sample_nums):
                available_sample_num += 1
                if self.random_mask is True:
                    random_mask = random.sample(range(0, self.input_dim), int(self.input_dim * self.mask_rate))
                    mask_arr = np.ones((1, self.input_dim))
                    mask_arr[0][random_mask] = 0
                else:
                    mask_arr = np.ones((1, self.input_dim))
                iter_num, x_adv_var, _attack_success = sess.run(self.launch_an_attack_substitute,
                                                                feed_dict={self.model.x_input: x_input_var,
                                                                           self.model.y_input: y_input_var,
                                                                           self.ForceIteration: self.force_iteration,
                                                                           self.random_mask_TS: mask_arr})
                if bool(_attack_success) is True:
                    iter_num_sum += iter_num
                    fool_num += 1

                logit = sess.run(self.model.softmax_output_logits, feed_dict={self.model.x_input: x_adv_var})[0]
                x_adv_all.append(x_adv_var[0])
                x_ori.append(x_input_var[0])
                logits.append(logit)

        x_adv_all = np.array(x_adv_all)
        x_ori = np.array(x_ori)
        logits = np.array(logits)
        logits_diff = np.mean(np.abs(logits[:, 0] - logits[:, 1]))
        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_all - x_ori) > 1e-6, axis=1))
        perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_all - x_ori), axis=1))
        """
        print(
            "perturb_l0 is {:.2f} and perturb_l1 is {:.2f} and average_adv_logits diff is {:.2f} and attack_success_rate "
            ": {:.2f}".format( 
                perturbations_amount_l0,
                perturbations_amount_l1,
                logits_diff,
                fool_num / available_sample_num * 100))
        """
        visualization_info = {}
        visualization_info['logits_diff'] = logits_diff
        visualization_info['perturbations_amount_l0'] = perturbations_amount_l0
        visualization_info['perturbations_amount_l1'] = perturbations_amount_l1
        visualization_info['attack_success_rate'] = fool_num / available_sample_num * 100
        visualization_info['average_iter_num'] = iter_num_sum / fool_num
        return x_adv_all, visualization_info

