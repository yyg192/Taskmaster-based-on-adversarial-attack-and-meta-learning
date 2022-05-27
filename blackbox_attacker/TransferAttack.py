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
        return (1 - acc) * 100


def _main():
    target_model_name = "target_model"
    substitute_model_name = "model_A"
    advtraining_method = "pgdl2"
    attack_method = "pgdl2"  # 或者是 "pgdl1" 或者是 "pgd_linfinity"
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
                                         is_trainable=True)  # 只能选True了，不然optimizer那里会报错
            target_model_load_dir = adv_train_root + "/" + target_model_name + "/adv" + str(
                advtraining_number)
            target_saver = tf.train.Saver()
            target_model_sess.run(tf.global_variables_initializer())
            #cur_checkpoint = tf.train.latest_checkpoint(target_model_load_dir)
            #target_saver = tf.train.import_meta_graph(cur_checkpoint+".meta")

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
            #cur_checkpoint = tf.train.latest_checkpoint(substitute_model_load_dir)
            #substitute_saver = tf.train.import_meta_graph(cur_checkpoint+".meta")
            substitute_model.load_param(substitute_model_load_dir, substitute_sess, substitute_saver)
            print("substitute_model_acc: ", substitute_sess.run(substitute_model.accuracy_output,
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
            attack_success_rate = TA.transfer_attack(malware_feature_vectors=malware_feature_vectors[:1000],
                                                     malware_feature_labels=malware_feature_labels[:1000],
                                                     substitute_sess=substitute_sess,
                                                     target_graph=target_model_graph,
                                                     target_sess=target_model_sess
                                                     )
            print("attack_success_rate: ", attack_success_rate)


if __name__ == "__main__":
    _main()
