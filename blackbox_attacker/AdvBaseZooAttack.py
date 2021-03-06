import tensorflow as tf
from tools.feature_reverser import DrebinFeatureReverse
from tools.file_operation import read_pickle
from models.teacher_model import teacher_model
from advtraining_methods.pgdl2_generator import pgdl2_generator
from advtraining_methods.pgdl1_generator import pgdl1_generator
from advtraining_methods.pgd_linfinity_generator import pgd_linfinity_generator
from config import config
from config import model_Param_dict
from keras.utils import to_categorical
import numpy as np
import tqdm
from attacker.AttackerBase import AttackerBase
from config import DREBIN_FEATURE_Param


class ZooAttack(AttackerBase):
    def __init__(self,
                 target_model,
                 epislon,
                 maximum_query_iterations=200):
        super(ZooAttack, self).__init__()
        self.target_model = target_model
        self.maximum_query_iterations = maximum_query_iterations
        self.epislon = epislon
        self.malware_label = tf.constant([0., 1.])
        self.n_dims = int(DREBIN_FEATURE_Param['feature_dimension'])
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_dims])
        self.scaled_max_extended_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.n_dims])
        self.scaled_min_extended_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.n_dims])
        self.rand_pos = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.last_prob = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.single_attack = self.single_iteration()

    def single_iteration(self):
        def f1(x, prob):
            return x, prob

        def f2(x2_clip, right_prob, x_adv_tmp, last_prob):  #
            def f3(x2, r_prob):
                return x2, r_prob

            def f4(x, las_prob):
                return x, las_prob

            return tf.cond(tf.less(right_prob[0], last_prob[0]),
                           lambda: f3(x2_clip, right_prob),
                           lambda: f4(x_adv_tmp, last_prob))

        diff = tf.one_hot(self.rand_pos, self.n_dims)
        # diff.shape = (1024, )
        diff = tf.multiply(diff, self.epislon)
        # diff.shape = (
        x1 = tf.subtract(self.x_input, diff)
        x1_clip = tf.clip_by_value(x1,
                                   clip_value_min=self.scaled_min_extended_TS,
                                   clip_value_max=self.scaled_max_extended_TS)
        left_prob = self.target_model.get_soft_output_logits(x1_clip)[:, 1]
        # left_prob.shape=(?, )
        x2 = tf.add(self.x_input, diff)
        x2_clip = tf.clip_by_value(x2,
                                   clip_value_min=self.scaled_min_extended_TS,
                                   clip_value_max=self.scaled_max_extended_TS)
        right_prob = self.target_model.get_soft_output_logits(x2_clip)[:, 1]
        x_adv_tmp, last_prob = tf.cond(tf.less(left_prob[0], self.last_prob[0]),
                                       lambda: f1(x1_clip, left_prob),
                                       lambda: f2(x2_clip, right_prob, self.x_input, self.last_prob))
        _attack_success = tf.less(last_prob, tf.constant(0.5))[0]
        debug_info = [left_prob, right_prob, diff]
        return _attack_success, x_adv_tmp, last_prob, debug_info

    def zoo_attack(self, malware_feature_vectors, epsilon=1., target_sess=None):
        attack_success_num = 0
        modified_num = 0
        query_totals = 0
        x_adv_all = []
        malware_feature_labels = to_categorical(np.ones(len(malware_feature_vectors)), num_classes=2)
        for single_malware_vector in tqdm.tqdm(malware_feature_vectors):
            x_adv_tmp = np.expand_dims(single_malware_vector, axis=0).copy()
            last_prob_init = target_sess.run(self.target_model.get_soft_output_logits(self.x_input),
                                             feed_dict={self.x_input: x_adv_tmp})

            las_prob = last_prob_init[:, 1]
            scaled_max_extended_var = np.maximum(
                np.multiply(self.scaled_clip_max, self.insertion_perm_array) +
                np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
                x_adv_tmp
            )
            scaled_min_extended_var = np.minimum(
                np.multiply(self.scaled_clip_min, self.removal_perm_array) +
                np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
                x_adv_tmp
            )

            search_domain = np.logical_or(scaled_min_extended_var, scaled_max_extended_var)
            search_domain_idx = np.where(search_domain[0] == True)[0]
            np.random.shuffle(search_domain_idx)

            for query_num in range(self.maximum_query_iterations):
                if query_num >= len(search_domain_idx):
                    x_adv_all.append(x_adv_tmp[0])
                    break
                rand_pos = np.array([search_domain_idx[query_num]])
                atss, x_adv_tmp, las_prob, debug_info = target_sess.run(self.single_attack,
                                                                        feed_dict={self.x_input: x_adv_tmp,
                                                                                   self.last_prob: las_prob,
                                                                                   self.scaled_min_extended_TS: scaled_min_extended_var,
                                                                                   self.scaled_max_extended_TS: scaled_max_extended_var,
                                                                                   self.rand_pos: rand_pos})
                if bool(atss) is True:
                    x_adv_all.append(x_adv_tmp[0])
                    attack_success_num += 1
                    modified_num += np.sum(np.abs(x_adv_tmp[0] - single_malware_vector) > 1e-6)
                    query_totals += (query_num + 1)
                    break
        print("attack_success_rate: ", attack_success_num / len(malware_feature_vectors) * 100)
        print("average_modified_num: ", modified_num / attack_success_num)
        print("average_query_num: ", query_totals / attack_success_num)
        return np.array(x_adv_all)


class TransferAttack:
    def __init__(self,
                 target_model,
                 substitute_model,
                 attacker):
        self.target_model = target_model
        self.substitute_model = substitute_model
        self.attacker = attacker

    def transfer_attack(self, malware_feature_vectors, malware_feature_labels,
                        substitute_sess, target_graph, target_sess):
        _, x_adv_all = \
            self.attacker.generate_attack_samples_teacher(malware_feature_vectors,
                                                          malware_feature_labels,
                                                          None,
                                                          substitute_sess)

        with target_graph.as_default():
            with target_sess.as_default():
                acc = target_sess.run(self.target_model.accuracy_output,
                                      feed_dict={self.target_model.x_input: x_adv_all,
                                                 self.target_model.y_input: malware_feature_labels})
        return x_adv_all, (1 - acc) * 100


def _main():
    ZOO_Iteration_nums = 1
    target_model_name = "target_model"
    substitute_model_name = "model_A"
    advtraining_method = "pgdl2"
    attack_method = "pgdl2"  # ????????? "pgdl1" ????????? "pgd_linfinity"
    substitute_advtraining_number = 0
    advtraining_number = 7
    malware_dataset_name = "virustotal_2018_5M_17M"
    benware_dataset_name = "androzoo_benware_3M_17M"
    exp_name = "3_4L_nobalanced_nobenwaretrain_15_epochs_4096_" + malware_dataset_name + "_AND_" + benware_dataset_name
    adv_train_root = config.get("advtraining.drebin", "advtraining_drebin_root") + "/" + advtraining_method + \
                     "_" + exp_name
    malware_feature_vectors = read_pickle(config.get(malware_dataset_name, "sample_vectors"))
    malware_feature_labels = np.ones(len(malware_feature_vectors))
    malware_feature_labels = to_categorical(malware_feature_labels, num_classes=2)
    target_model_graph = tf.Graph()
    target_model_sess = tf.Session(graph=target_model_graph)
    substitute_graph = tf.Graph()
    substitute_sess = tf.Session(graph=substitute_graph)
    with target_model_graph.as_default():
        with target_model_sess.as_default():
            target_model = teacher_model(hyper_parameter=model_Param_dict[target_model_name],
                                         model_name=target_model_name,
                                         is_trainable=True)  # ?????????True????????????optimizer???????????????
            ZooA = ZooAttack(target_model, 1, DREBIN_FEATURE_Param['feature_dimension'])
            target_model_load_dir = adv_train_root + "/" + target_model_name + "/adv" + str(
                advtraining_number)
            # target_saver = tf.train.Saver()
            target_model_sess.run(tf.global_variables_initializer())
            # cur_checkpoint = tf.train.latest_checkpoint(target_model_load_dir)
            target_saver = tf.train.Saver()  # tf.train.import_meta_graph(cur_checkpoint+".meta")
            target_model.load_param(target_model_load_dir, target_model_sess, target_saver)
            print("target_model_acc: ", target_model_sess.run(target_model.accuracy_output,
                                                              feed_dict={target_model.x_input: malware_feature_vectors,
                                                                         target_model.y_input: malware_feature_labels
                                                                         }))

    with substitute_graph.as_default():
        with substitute_sess.as_default():
            substitute_model = teacher_model(hyper_parameter=model_Param_dict[substitute_model_name],
                                             model_name=substitute_model_name,
                                             is_trainable=True)
            substitute_model_load_dir = adv_train_root + "/" + substitute_model_name + "/adv" + str(
                substitute_advtraining_number)
            substitute_sess.run(tf.global_variables_initializer())
            substitute_saver = tf.train.Saver()
            # substitute_saver = tf.train.import_meta_graph(cur_checkpoint+".meta")
            substitute_model.load_param(substitute_model_load_dir, substitute_sess, substitute_saver)
            print("substitute_acc: ", substitute_sess.run(substitute_model.accuracy_output,
                                                          feed_dict={substitute_model.x_input: malware_feature_vectors,
                                                                     substitute_model.y_input: malware_feature_labels
                                                                     }))
            if attack_method is "pgdl2":
                attacker = pgdl2_generator(target_model=substitute_model,
                                           maximum_iterations=10,
                                           force_iteration=False,
                                           use_search_domain=True,
                                           random_mask=False,
                                           mask_rate=0.2,
                                           step_size=10.,
                                           )
            elif attack_method is "pgdl1":
                attacker = pgdl1_generator(target_model=substitute_model,
                                           maximum_iterations=50,
                                           force_iteration=True,
                                           use_search_domain=True,
                                           random_mask=False,
                                           mask_rate=0.4,
                                           top_k=1,
                                           step_size=1.
                                           )
            elif attack_method is "pgd_linfinity":
                attacker = pgd_linfinity_generator(target_model=substitute_model,
                                                   maximum_iterations=100,
                                                   force_iteration=False,
                                                   use_search_domain=True,
                                                   random_mask=False,
                                                   mask_rate=0.3,
                                                   step_size=0.01)

            TA = TransferAttack(target_model=target_model,
                                substitute_model=substitute_model,
                                attacker=attacker
                                )
            x_adv_all, attack_success_rate = TA.transfer_attack(malware_feature_vectors=malware_feature_vectors[:500],
                                                                malware_feature_labels=malware_feature_labels[:500],
                                                                substitute_sess=substitute_sess,
                                                                target_graph=target_model_graph,
                                                                target_sess=target_model_sess
                                                                )
            print("#################### Transfer Attak ######################")
            print("attack_success_rate: ", attack_success_rate)

            print("#################### Adv Base Zoo Attack ###################")
            with target_model_graph.as_default():
                with target_model_sess.as_default():
                    for i in range(ZOO_Iteration_nums):
                        x_adv_all = ZooA.zoo_attack(x_adv_all, epsilon=1, target_sess=target_model_sess)


if __name__ == "__main__":
    _main()
