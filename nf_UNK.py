import os, glob, re,  sys, argparse, time, random
from random import shuffle
import tensorflow as tf
import numpy as np
from PIL import Image
from nf_UTILS import *
from nf_MODEL import model
#from tensorflow.contrib.slim.nets import resnet_v1
#import tensorflow.contrib.slim as slim

tf.logging.set_verbosity(tf.logging.WARN)

EXP_DATA = '20190601-av1-2k-sepGrad-small'
TRAIN_DATA_PATH = "/home/chenjs/nf/data/train/small_low"
TRAIN_LABEL_PATH = "/home/chenjs/nf/data/train/small_high"
VALID_DATA_PATH = "/home/chenjs/ee-new/data/test/18/av1_18_qp53"
VALID_LABEL_PATH = "/home/chenjs/ee-new/data/test/18/gt"
#VALID_DATA_PATH = ''
#VALID_LABEL_PATH = ''
LOG_PATH = "./logs/%s/"%(EXP_DATA)
CKPT_PATH = "./checkpoints/%s/"%(EXP_DATA)
DEVIDE_SIZE = 0
PATCH_SIZE = (64,64)
BATCH_SIZE = 64
BASE_LR = 0.0001
LR_DECAY_RATE = 0.5
LR_DECAY_STEP = 50
MAX_EPOCH = 600
IMAGE_BATCH = 4

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

'''
ph : Place Holder
so : Sess Output

'''
if __name__ == '__main__':

    #===================prepare data==================#
    #The following variables contain the whole training/validation set.
    #They contain a large number of [imagedate, label]
    train_data = get_train_pair(TRAIN_DATA_PATH, TRAIN_LABEL_PATH)
    #train_data = get_train_list(load_file_list(TRAIN_DATA_PATH), load_file_list(TRAIN_LABEL_PATH))

    len_train_data = len(train_data)
    #print("num of train data:%d"%(len_train_data))
    stepsPerEpoch = len_train_data // BATCH_SIZE

    if (VALID_LABEL_PATH and VALID_DATA_PATH):
        valid_data = get_train_pair(VALID_DATA_PATH, VALID_LABEL_PATH)
        phValInput = tf.placeholder('float32', shape=(1, None, None, 1))
        phValGt    = tf.placeholder('float32', shape=(1, None, None, 1))

        valOutput = tf.multiply(tf.clip_by_value(model(phValInput), 0., 1.), 255)


    #==================define model==================#
    with tf.name_scope('input_scope'):
        phTrainInput = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        phTrainGt = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

        trainOutput = tf.multiply(tf.clip_by_value(model(phTrainInput), 0., 1.), 255)

    with tf.name_scope('loss_scope'):

        loss = tf.reduce_mean(tf.square(tf.subtract(trainOutput, phTrainGt)))
        #loss = tf.reduce_sum(tf.square(tf.subtract(trainOutput, phTrainGt)))
        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
        loss += tf.add_n(weights) * 1e-4

        avg_loss = tf.placeholder('float32')
        tf.summary.scalar("avg_loss", avg_loss)


    with tf.name_scope('optimization'):
        global_step     = tf.Variable(0, trainable=False)
        learning_rate   = tf.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP*stepsPerEpoch, LR_DECAY_RATE, staircase=True)
        tf.summary.scalar("learning rate", learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate, 0.9)
        opt = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    #========================start training==================#
    with tf.Session(config=config) as sess:

        makeDirsIfNotExist(LOG_PATH)
        makeDirsIfNotExist(CKPT_PATH)


        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        sess.run(tf.global_variables_initializer())

        if model_path:
            print("restore model...")
            saver.restore(sess, model_path)
            print("Done")

        for epoch in range(MAX_EPOCH):
            shuffle(train_data)
            total_g_loss, n_iter = 0, 0

            start_time = time.time()

            for idx in range(stepsPerEpoch):
                input_data, gt_data = get_batch_data(train_data, BATCH_SIZE, idx)
                #input_data, gt_data, cbcr_data = prepare_nn_data(train_data, idx)
                feed_dict = {phTrainInput: input_data, phTrainGt: gt_data}
                _, soLoss, soTrainOutput, g_step= sess.run([opt, loss, trainOutput, global_step], feed_dict=feed_dict)

                total_g_loss += soLoss
                n_iter += 1

            lr, summary = sess.run([learning_rate, merged], {avg_loss:total_g_loss/n_iter})
            file_writer.add_summary(summary, epoch)

            epoch_time = time.time() - start_time
            avgLoss = total_g_loss/n_iter
            tf.logging.warning("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f"%(epoch, MAX_EPOCH, epoch_time, avgLoss, lr))

            #There is no need to save ckpt for every epoch; besides it costs much space.
            if ((epoch+1) % 10 == 0):
                saver.save(sess, os.path.join(CKPT_PATH, "%s_%03d.ckpt"%(EXP_DATA, epoch)))

                #=======validation========#
                #It ought to be a geninue validation set, but here directly uses test set for the sack of convience, because this test set is relatively small.
                if (VALID_LABEL_PATH and VALID_DATA_PATH):
                    sumPsnr = 0
                    numValidData = len(valid_data)
                    for i in range(numValidData):
                        valid_input = np.reshape(valid_data[i][0], (1, valid_data[i][0].shape[0], valid_data[i][0].shape[1], valid_data[i][0].shape[2]))
                        soValidOutput = sess.run(valOutput, feed_dict={phValInput:valid_input})
                        sumPsnr += psnr(valid_data[i][1], soValidOutput)
                    avgPsnr = sumPsnr / numValidData
                    tf.logging.warning("Epoch: %d, avgPsnrOnValid: %.4f"%(epoch, avgPsnr))

