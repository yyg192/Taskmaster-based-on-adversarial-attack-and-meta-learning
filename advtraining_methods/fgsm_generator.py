from tools import utils
import numpy as np
from attacker.AttackerBase import AttackerBase
import tensorflow as tf
import tqdm
import random


class fgsm_generator(AttackerBase):
    def __init__(self,
                 target_model):
        super(fgsm_generator, self).__init__()
        self.model = target_model
        self.scaled_clip_min_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.scaled_clip_max_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.random_mask_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.batch_size = 1
        self.ord = "l2"
        self.epsilon = 6.
        self.mask_rate = 0.3
        self.launch_an_attack = self.graph()
        self.use_search_domain = False

    def graph(self):

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
        perturbations = tf.multiply(perturbations, self.random_mask_TS)
        x_adv_tmp = self.model.x_input + perturbations
        x_adv_tmp_clip = tf.clip_by_value(x_adv_tmp,
                                          clip_value_min=self.scaled_clip_min_TS,
                                          clip_value_max=self.scaled_clip_max_TS)

        return x_adv_tmp_clip

    def generate_attack_samples(self,
                                attack_feature_vectors,
                                attack_feature_labels,
                                require_sample_nums,
                                sess):
        """
        attack_feature_vectors只能是一个样本！
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
        for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
            available_sample_num += 1
            if self.use_search_domain is True:
                scaled_max_extended = np.maximum(
                    np.multiply(self.scaled_clip_max,
                                self.insertion_perm_array) +  # upper bound for positions allowing perturbations
                    np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
                    # may be useful to reset the lower bound
                    x_input_var  # upper bound for positions no perturbations allowed
                )
                scaled_min_extended = np.minimum(
                    np.multiply(self.scaled_clip_min, self.removal_perm_array) +
                    np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
                    x_input_var
                )
            else:
                scaled_max_extended = self.scaled_clip_max
                scaled_min_extended = self.scaled_clip_min
            for _ in range(require_sample_nums):
                random_mask = random.sample(range(0, self.input_dim), int(self.input_dim * self.mask_rate))
                mask_arr = np.ones((1, self.input_dim))
                mask_arr[0][random_mask] = 0
                x_adv_var, perturbations = sess.run(self.launch_an_attack,
                                                    feed_dict={self.model.x_input: x_input_var,
                                                               self.model.y_input: y_input_var,
                                                               self.scaled_clip_min_TS: scaled_min_extended,
                                                               self.scaled_clip_max_TS: scaled_max_extended,
                                                               self.random_mask_TS: mask_arr})

                logit = sess.run(self.model.softmax_output_logits, feed_dict={self.model.x_input: x_adv_var})[0]
                x_adv_all.append(x_adv_var[0])
                x_ori.append(x_input_var[0])
                logits.append(logit)

        x_adv_all = np.array(x_adv_all)
        x_ori = np.array(x_ori)
        logits = np.array(logits)
        avg_logits = np.mean(logits, axis=0)
        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_all - x_ori) > 1e-6, axis=1))
        perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_all - x_ori), axis=1))

        print("perturb_l0 is {:.2f} and perturb_l1 is {:.2f} and average_adv_logits is {:.2f}".format(perturbations_amount_l0,
                                                                                                      perturbations_amount_l1,
                                                                                                      avg_logits))
        return x_adv_all
