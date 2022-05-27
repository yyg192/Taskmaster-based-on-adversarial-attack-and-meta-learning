from tools.utils import read_pickle
from sklearn.model_selection import train_test_split
import tqdm
from config import DREBIN_FEATURE_Param
import numpy as np
import tensorflow as tf
from config import config
from config import model_Param_dict
from models.teacher_model import teacher_model
import time
from keras.utils import to_categorical
from tools.feature_reverser import DrebinFeatureReverse
from tools import utils

"""
code is from  https://github.com/Xiang-cd/realsafe/blob/39f632e950/realsafe/
"""


def model_pred_class(model, sess, x_input):
    target_pred = sess.run(model.y_pred_output,
                           feed_dict={model.x_input: x_input})
    return target_pred[0]


def model_get_loss(model, sess, eval_points):
    malware_labels = to_categorical(np.ones((len(eval_points))), num_classes=2)

    xent_loss = sess.run(model.xent_cross_entropy,
                         feed_dict={model.x_input: eval_points,
                                    model.y_input: malware_labels})
    return xent_loss


def nes_attack(sess, args, model, initial_img, IMAGE_SIZE):
    plot_ite = 1000
    max_lr = args["max_lr"]
    max_iters = int(np.ceil(args["max_queries"] // args["samples_per_draw"]))
    lower, upper, normalizer = get_scaled_min_max_extended(initial_img)

    adv = initial_img.copy()

    # assert orig_class == model.pred_class(initial_img, axis=0), 'input image must be correctly classified'
    print('predicted class %d' % model_pred_class(model, sess, initial_img))

    # HISTORY VARIABLES (for backtracking and momentum)
    num_queries = 0
    g = 0

    # adv = np.expand_dims(adv, axis=0) # wrap(unsqueeze) image to ensure 4-D np.array
    # target_class 的形状可以是 (num_imgs,) 或 (num_imgs, num_class) 并自由选择适合新黑盒攻击的形状。
    last_ls = []
    x_s = []
    # BEGIN ATTACK
    # MAIN LOOP
    for i in range(max_iters):
        # record the intermediate attack results
        x_s.append(adv)
        ##  ----------start the attack below-------- ##
        start = time.time()
        # GET GRADIENT
        prev_g = g
        l, g = get_grad_np(sess, args, model, adv, args["samples_per_draw"], args["nes_batch_size"], IMAGE_SIZE, lower, upper, normalizer)
        # print(l, g.shape)
        # SIMPLE MOMENTUM
        g = args["momentum"] * prev_g + (1.0 - args["momentum"]) * g
        # CALCULATE PROBABILITY
        # eval_probs_val = model.predict_prob(adv)
        # CHECK IF WE SHOULD STOP

        pred_class = model_pred_class(model, sess, adv)[0]
        if pred_class == 0:  # 如果是预测为良性软件了！
            break
        """
        padv = model.eval_adv(sess, adv, target_class) #这里自己改一下就好了，如果攻击成功则停止
        predicted_class = model_pred_class(model, sess, adv)
        if (padv == 1 and is_targeted == 1) or (padv == 0 and is_targeted == -1):  # and epsilon <= goal_epsilon:
            print('[log] early stopping at iteration {}'.format(i))
            print('[Final][Info]:predicted class: {}}, target class: {})'.format(predicted_class, target_class))
            break
        """
        # PLATEAU LR ANNEALING (if loss trend decreases in plateau_length,  then max_lr anneals)
        last_ls.append(l)
        last_ls = last_ls[-args["plateau_length"]:]
        if last_ls[-1] > last_ls[0] and len(last_ls) == args["plateau_length"]:
            if max_lr > args["min_lr"]:
                print("[log] Annealing max_lr")
                max_lr = max(max_lr / args["plateau_drop"], args["min_lr"])
            last_ls = []

        # ATTACK
        current_lr = max_lr
        proposed_adv = adv + current_lr * np.sign(g)  # (1,32,32,3) - (32,32,3) = (1,32,32,3)
        proposed_adv = np.clip(proposed_adv, lower, upper)
        prev_adv = adv
        adv = proposed_adv
        # BOOK-KEEPING STUFF
        num_queries += args["samples_per_draw"]
        predicted_class = model_pred_class(model, sess,
                                           adv)  ##**************************************************************
        if (i + 1) % plot_ite == 0:
            print('Step {}: number of query {}}, loss {:8f}, lr {:.2E}, '
                  'predicted class {} (time {:.4f})'
                  .format(i, num_queries, l, current_lr, predicted_class, time.time() - start))

    return x_s, num_queries, adv


def get_grad_np_old(sess, args, model, adv, spd, bs, IMAGE_SIZE):
    # tc就是target_class也就是-1的意思 bs是batch_size，外面设置了bs=50，IMAGE_SIZE=1024
    num_batches = spd // bs  # 每个batch产生多少个samples噪音点
    losses_val = []
    grads_val = []
    # 假如batch_size=1的话，num_batches就有50了
    for _ in range(num_batches):
        noise_pos = np.random.normal(size=(bs // 2, + IMAGE_SIZE))  # (25, 1024)
        noise = np.concatenate([noise_pos, -noise_pos], axis=0)  # (50,1024) ui
        eval_points = adv + args["sigma"] * noise  # (50, 1024)  x +- ui
        #########################在这里还要clip一下顺便还要utils.map_to_discrete_domain一下！！！
        loss_val = model_get_loss(model, sess, eval_points)  # 推测(batch_size, )
        # loss_val值只有一个吗？
        losses_tiled = np.tile(np.reshape(loss_val, (-1, 1)), np.prod(IMAGE_SIZE))  # 推测(batch_size, IMAGE_SIZE)
        # losses_tiled = np.reshape(losses_tiled, (bs,) + IMAGE_SIZE)
        grad_val = np.mean(losses_tiled * noise, axis=0) / args["sigma"]
        losses_val.append(loss_val)
        grads_val.append(grad_val)
    return np.array(losses_val).mean(), np.mean(np.array(grads_val), axis=0)

"""
def get_grad_np(sess, args, model, adv, spd, bs, IMAGE_SIZE, lower, upper, normalizer):
    # tc就是target_class也就是-1的意思 bs是batch_size，外面设置了bs=50，IMAGE_SIZE=1024
    # num_batches = spd // bs # 每个batch产生多少个samples噪音点
    losses_val = []
    grads_val = []
    # 假如batch_size=1的话，num_batches就有50了
    for _ in range(bs):  # adv.shape = (1,1024)
        noise_pos = np.random.normal(size=(spd // 2, IMAGE_SIZE))  # (25, 1024)
        noise = np.concatenate([noise_pos, -noise_pos], axis=0)  # (50,1024) ui
        eval_points = adv + args["sigma"] * noise  # eval_points: (50, 1024)  x +- ui
        eval_points = np.clip(eval_points, lower, upper)
        eval_points = utils.map_to_discrete_domain(normalizer, eval_points)
        #########################在这里还要clip一下顺便还要utils.map_to_discrete_domain一下！！！
        loss_val = model_get_loss(model, sess, eval_points)  # 推测(batch_size, )
        # loss_val值只有一个吗？
        losses_tiled = np.tile(np.reshape(loss_val, (-1, 1)), np.prod(IMAGE_SIZE))  # 推测(batch_size, IMAGE_SIZE)
        # losses_tiled = np.reshape(losses_tiled, (bs,) + IMAGE_SIZE)
        grad_val = np.mean(losses_tiled * noise, axis=0) / args["sigma"]
        losses_val.append(loss_val)
        grads_val.append(grad_val)
    return np.array(losses_val).mean(), np.mean(np.array(grads_val), axis=0)
"""

def get_scaled_min_max_extended(x_input):
    feature_reverser = DrebinFeatureReverse()
    normalizer = feature_reverser.normalizer
    clip_min, clip_max = utils.get_min_max_bound(normalizer=normalizer)
    scaled_clip_min = utils.normalize_transform(np.reshape(clip_min, (1, -1)), normalizer=normalizer)
    scaled_clip_max = utils.normalize_transform(np.reshape(clip_max, (1, -1)), normalizer=normalizer)
    insertion_perm_array, removal_perm_array = feature_reverser.get_mod_array()
    _scaled_max_extended_var = np.maximum(
        np.multiply(scaled_clip_max, insertion_perm_array) +
        np.multiply(scaled_clip_min, 1. - insertion_perm_array),
        x_input
    )
    _scaled_min_extended_var = np.minimum(
        np.multiply(scaled_clip_min, removal_perm_array) +
        np.multiply(scaled_clip_max, 1. - removal_perm_array),
        x_input
    )
    return _scaled_min_extended_var, _scaled_max_extended_var, normalizer


"""linf"""
args = {
    "samples_per_draw": 5,
    "nes_batch_size": 1,
    "sigma": 1,
    "epsilon": 0.5,
    "momentum": 0.9,
    "max_queries": 200,
    "plateau_drop": 2.0,
    "plateau_length": 5,
    "max_lr": 1e-2,
    "min_lr": 0.01275,
}

if __name__ == "__main__":
    target_model_name = "target_model"
    advtraining_number = 0
    advtraining_method = "pgdl2"
    malware_dataset_name = "virustotal_2018_5M_17M"
    benware_dataset_name = "androzoo_benware_3M_17M"
    exp_name = "4096_" + malware_dataset_name + "_AND_" + benware_dataset_name
    adv_train_root = config.get("advtraining.drebin", "advtraining_drebin_root") + "/" + advtraining_method + \
        "_" + exp_name
    ori_malware_feature_vectors = read_pickle(config.get(malware_dataset_name, "sample_vectors"))
    ori_malware_feature_labels = np.ones(len(ori_malware_feature_vectors))
    ori_malware_feature_labels = to_categorical(ori_malware_feature_labels, num_classes=2)
    target_graph = tf.Graph()
    target_sess = tf.Session(graph=target_graph)
    with target_graph.as_default():
        with target_sess.as_default():
            target_model = teacher_model(hyper_parameter=model_Param_dict[target_model_name],
                                         model_name=target_model_name,
                                         is_trainable=True)
            target_model_load_dir = adv_train_root + "/" + target_model_name + "/adv" + str(
                advtraining_number)
            target_sess.run(tf.global_variables_initializer())
            target_saver = tf.train.Saver()
            target_model.load_param(target_model_load_dir, target_sess, target_saver)
            result = target_sess.run(target_model.accuracy_output,
                                  feed_dict={target_model.x_input: ori_malware_feature_vectors,
                                             target_model.y_input: ori_malware_feature_labels})
            print("target_model_accuracy: {}".format(result))
            for single_malware_feature_vector in ori_malware_feature_vectors:
                nes_attack(sess=target_sess, args=args, model=target_model,
                           initial_img=np.array([single_malware_feature_vector]).copy(),
                           IMAGE_SIZE=DREBIN_FEATURE_Param['feature_dimension'])
# (sess, args, model, attack_seed, initial_img, target_class, class_num, IMAGE_SIZE):
