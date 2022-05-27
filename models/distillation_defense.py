import tensorflow as tf
from tools.DataProducer import DataProducer2
import numpy as np
from keras.utils import to_categorical
import os
from timeit import default_timer
from tools.file_operation import read_pickle
from sklearn.model_selection import train_test_split
from tools import utils
from config import config
from config import model_Param_dict
from models.teacher_model import teacher_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


class distillation_model:
    def __init__(self,
                 hyper_parameter,
                 model_name,
                 temperature,
                 alpha,
                 is_trainable=False,
                 reuse=False,
                 ):
        """
        变量接收区
        """
        self.model_name = model_name
        self.hp_params = utils.ParamWrapper(hyper_parameter)
        self.is_trainable = is_trainable
        self.T = temperature
        self.alpha = alpha

        """
        执行区
        """
        self.nn = BASIC_DNN_GRAPH  ##### 注意，我们可以通过修改graph函数来更改我们的模型 #####
        self.model_graph(reuse=reuse)
        self.global_train_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = \
                tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.loss,
                                                                              global_step=self.global_train_step)
    def model_graph(self, reuse=False):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.hp_params.input_dim], name='X')
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[None, self.hp_params.output_dim], name='Y')
        self.soft_y_input = tf.placeholder(dtype=tf.float32, shape=[None, self.hp_params.output_dim], name='Y_soft')
        self.is_backbone_training = tf.placeholder(dtype=tf.bool, name="BACKBONE_TRAINING")  # 这个用于dropout和batchnorm的
        random_seed = np.random.randint(low=0, high=9999)
        tf.set_random_seed(random_seed)
        self.output_logits = self.nn(self.x_input,
                                     name=self.model_name,
                                     hidden_neurons=self.hp_params.hidden_neurons,
                                     output_dim=self.hp_params.output_dim,
                                     trainable=self.is_trainable,
                                     reuse=reuse)
        self.student_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits,
                                                                    labels=self.y_input)
        KL_fn = tf.keras.losses.KLDivergence(reduction='none')
        self.distillation_loss = KL_fn(
            tf.nn.softmax(self.soft_y_input/self.T, axis=1),
            tf.nn.softmax(self.output_logits/self.T, axis=1)
        )
        self.loss = self.alpha*self.student_loss + (1-self.alpha)*self.distillation_loss
        self.soft_output_logits = tf.nn.softmax(self.output_logits)
        self.hard_output_logits = tf.one_hot(tf.argmax(self.soft_output_logits, axis=1),
                                             self.hp_params.output_dim)


        self.y_pred_output = tf.argmax(self.output_logits, axis=1, output_type=tf.int32)

        self.accuracy_output = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred_output, tf.argmax(self.y_input, axis=-1, output_type=tf.int32)))
        )


    def get_output_logits(self, x_tensor):
        fc_output_logits = self.nn(x_input=x_tensor,
                                   name=self.model_name,
                                   hidden_neurons=self.hp_params.hidden_neurons,
                                   output_dim=self.hp_params.output_dim,
                                   trainable=False,
                                   reuse=True)
        return fc_output_logits

    def get_output_pred(self, x_tensor):
        output_logits = self.get_output_logits(x_tensor)
        y_pred_output = tf.argmax(output_logits, axis=1, output_type=tf.int32)
        return y_pred_output

    def get_soft_output_logits(self, x_tensor):
        fc_output_logits = self.nn(x_input=x_tensor,
                                   name=self.model_name,
                                   hidden_neurons=self.hp_params.hidden_neurons,
                                   output_dim=self.hp_params.output_dim,
                                   trainable=False,
                                   reuse=True)
        softmax_output_logits = tf.nn.softmax(fc_output_logits)
        return softmax_output_logits

    def get_hard_output_logits(self, x_tensor):
        fc_output_logits = self.nn(x_input=x_tensor,
                                   name=self.model_name,
                                   hidden_neurons=self.hp_params.hidden_neurons,
                                   output_dim=self.hp_params.output_dim,
                                   trainable=False,
                                   reuse=True)
        hard_logits = tf.one_hot(tf.argmax(fc_output_logits, axis=1), self.hp_params.output_dim)
        return hard_logits

    def load_param(self, load_dir, sess, saver):
        if load_dir is None:
            raise ValueError("the load_dir is None, please check your code!")
        cur_checkpoint = tf.train.latest_checkpoint(load_dir)
        if cur_checkpoint is None:
            raise ValueError("the content of your checkpoint file is wrong!!!")
        #meta_dir = cur_checkpoint + ".meta"
        #restorer = tf.train.import_meta_graph(meta_dir)
        saver.restore(sess, cur_checkpoint)

    def train_backbone(self, train_x, train_y, train_soft_y, valid_x, valid_y, valid_soft_y,
                       sess,
                       saver,
                       load_checkpoint_dir=None,
                       save_checkpoint_dir=None):
        train_input = DataProducer2(train_x, train_y, train_soft_y, self.hp_params.batch_size, n_epochs=10)
        valid_input = DataProducer2(valid_x, valid_y, valid_soft_y, self.hp_params.batch_size, n_epochs=10)

        if load_checkpoint_dir is not None:
            cur_checkpoint = tf.train.latest_checkpoint(load_checkpoint_dir)
            saver.restore(sess, cur_checkpoint)
        training_time = 0.0
        train_input.reset_cursor()
        output_steps = 100  ##############################################################################
        for step_idx, X_train_batch, Y_train_batch, Y_train_soft_batch in train_input.next_batch():
            train_dict = {
                self.x_input: X_train_batch,
                self.y_input: Y_train_batch,
                self.soft_y_input: Y_train_soft_batch
            }
            if (step_idx + 1) % output_steps == 0:
                print('Step {}/{}'.format(step_idx + 1, train_input.steps))
                valid_input.reset_cursor()
                valid_result_list = [sess.run([self.accuracy_output, self.y_pred_output],
                                              feed_dict={self.x_input: X_valid_batch,
                                                         self.y_input: Y_valid_batch
                                                         })
                                     for [_, X_valid_batch, Y_valid_batch, _] in valid_input.next_batch()
                                     ]
                valid_result = np.array(valid_result_list)
                _acc = np.mean(valid_result[:, 0])
                _pred_y = np.concatenate(valid_result[:, 1])
                from sklearn.metrics import f1_score
                _f1_score = f1_score(np.argmax(valid_y, axis=-1), _pred_y[:valid_y.shape[0]])
                print('    validation accuracy {:.5}%'.format(_acc * 100))
                print('    validation f1 score {:.5}%'.format(_f1_score * 100))
            start = default_timer()
            sess.run(self.optimizer, feed_dict=train_dict)
            end = default_timer()
            training_time = training_time + end - start

        if save_checkpoint_dir is not None:
            if os.path.exists(save_checkpoint_dir) is None:
                os.makedirs(save_checkpoint_dir)
            saver.save(sess, save_checkpoint_dir+'/checkpoint', global_step=self.global_train_step)
            print("save backbone model")
        else:
            print("without saving the backbone model")


def _main():
    advtraining_number = 1
    malware_dataset_name = "virustotal_2018_5M_17M"
    benware_dataset_name = "androzoo_benware_3M_17M"
    advtraining_method = "pgdl2"
    exp_name = "4096_" + malware_dataset_name + "_AND_" + benware_dataset_name
    adv_train_root = config.get("advtraining.drebin", "advtraining_drebin_root") + "/" + advtraining_method + \
                     "_" + exp_name
    ori_malware_features_vectors = read_pickle(config.get(malware_dataset_name, "sample_vectors"))
    ori_benware_features_vectors = read_pickle(config.get(benware_dataset_name, "sample_vectors"))
    min_len = min(len(ori_malware_features_vectors), len(ori_benware_features_vectors)) - 1
    ori_malware_features_vectors, _ = \
        train_test_split(ori_malware_features_vectors,
                         train_size=min_len - 1,
                         random_state=np.random.randint(0, 999))
    ori_benware_features_vectors, _ = \
        train_test_split(ori_benware_features_vectors,
                         train_size=min_len - 1,
                         random_state=np.random.randint(0, 999))
    ori_malware_features_labels = to_categorical(np.ones(min_len - 1), num_classes=2)
    ori_benware_features_labels = to_categorical(np.zeros(min_len - 1), num_classes=2)
    all_features_vectors = np.concatenate((ori_malware_features_vectors, ori_benware_features_vectors), axis=0)
    all_features_labels = np.concatenate((ori_malware_features_labels, ori_benware_features_labels), axis=0)
    target_graph = tf.Graph()
    target_sess = tf.Session(graph=target_graph)

    student_graph = tf.Graph()
    student_sess = tf.Session(graph=student_graph)

    with target_graph.as_default():
        with target_sess.as_default():
            target_model = teacher_model(hyper_parameter=model_Param_dict['target_model'],
                                         model_name='target_model',
                                         is_trainable=True)
            target_model_load_dir = adv_train_root + "/target_model/adv" + str(
                advtraining_number)
            target_saver = tf.train.Saver()
            target_sess.run(tf.global_variables_initializer())
            target_model.load_param(target_model_load_dir, target_sess, target_saver)
            soft_labels = target_sess.run(target_model.soft_output_logits,
                                          feed_dict={target_model.x_input: all_features_vectors})

    with student_graph.as_default():
        with student_sess.as_default():
            student_model = distillation_model(hyper_parameter=model_Param_dict['target_model'],
                                               model_name='student_model',
                                               temperature=10,
                                               alpha=0.1,
                                               is_trainable=True)
            student_saver = tf.train.Saver()
            student_sess.run(tf.global_variables_initializer())
            train_features, valid_features, train_y, valid_y, train_soft_y, valid_soft_y  = \
                train_test_split(all_features_vectors, all_features_labels, soft_labels,
                                 test_size=0.1, random_state=np.random.randint(0, 999))
            save_dir = config.get("DEFAULT", "project_root")+"/distillation_model_checkpoint"
            student_model.train_backbone(train_x=train_features, train_y=train_y,
                                         valid_x=valid_features, valid_y=valid_y,
                                         train_soft_y=train_soft_y, valid_soft_y=valid_soft_y,
                                         sess=student_sess,
                                         saver=student_saver,
                                         save_checkpoint_dir=save_dir)


if __name__ == "__main__":
    _main()


