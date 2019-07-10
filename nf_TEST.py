import numpy as np
from PIL import Image
import tensorflow as tf
import os, time
from nf_MODEL import model
#from VDSRMODELold import model
from nf_UTILS import *

tf.logging.set_verbosity(tf.logging.WARN)

EXP_DATA = "20190601-av1-2k-sepGrad-big"
TESTOUT_PATH = "./testout/%s/"%(EXP_DATA)
MODEL_PATH = "./checkpoints/%s/"%(EXP_DATA)
#MODEL_PATH = "/home/chenjs/a4/chenjs/checkpoints_before_chen/in-loop-filter/0611"
ORIGINAL_PATH = "/home/chenjs/ee-new/data/test/18/av1_18_qp53"
GT_PATH = "/home/chenjs/ee-new/data/test/18/gt"
#ORIGINAL_PATH = "/home/chenjs/relHM/sky_rebuild"
#GT_PATH = "/home/chenjs/relHM/sky_true"
OUT_DATA_PATH = "./outdata/%s/"%(EXP_DATA)
NOFILTER = {'qp22':41.7929, 'qp27':37.6837, 'qp32':33.9330, 'qp37':30.5175}

##Ground truth images dir should be the 2nd component of 'fileOrDir' if 2 components are given.

##cb, cr components are not implemented
def prepare_test_data(fileOrDir):
    if not os.path.exists(TESTOUT_PATH):
        os.mkdir(TESTOUT_PATH)

    original_ycbcr = []
    gt_y = []
    fileName_list = []
    #The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        #w, h = getWH(fileOrDir)
        #imgY = getYdata(fileOrDir, [w, h])
        imgY = c_getYdata(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            #w, h = getWH(path)
            #imgY = getYdata(path, [w, h])
            imgY = c_getYdata(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:

            or_imgY = c_getYdata(pair[0])
            gt_imgY = c_getYdata(pair[1])

            #normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1],1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1],1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list

def test_all_ckpt(modelPath, fileOrDir):
    max = [0, 0]

    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])

    re_psnr = tf.placeholder('float32')
    tf.summary.scalar('re_psnr', re_psnr)

    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        #shared_model = tf.make_template('shared_model', model)
        output_tensor = model(input_tensor)
        #output_tensor, weight = shared_model(input_tensor)
        output_tensor = tf.clip_by_value(output_tensor, 0., 1.)
        output_tensor = output_tensor * 255

        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(OUT_DATA_PATH, sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())


        original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)

        for ckpt in ckptFiles:
            epoch = int(ckpt.split('_')[1].split('.')[0])
            if epoch < 0:
                continue

            saver.restore(sess,os.path.join(modelPath,ckpt))
            total_time, total_psnr = 0, 0
            total_imgs = len(fileName_list)
            for i in range(total_imgs):
                imgY = original_ycbcr[i][0]
                #imgCbCr = original_ycbcr[i][1]
                gtY = gt_y[i] if gt_y else 0
                #gtY = gtY * 255

                start_t = time.time()
                out = sess.run(output_tensor, feed_dict={input_tensor: imgY})

                duration_t = time.time() - start_t
                total_time += duration_t

                # save_path = os.path.join(TESTOUT_PATH, str(epoch), os.path.basename(fileName_list[i]))
                # if not os.path.exists(os.path.dirname(save_path)):
                #     os.makedirs(os.path.dirname(save_path))

                # save_test_img(out, imgCbCr, save_path)

                ## gt_y is not empty means 'ground truth' is offered
                if gt_y:
                    p = psnr(out, gtY)
                    total_psnr += p
                    print("filename:%s, psnr:%.4f"%(fileName_list[i], p))
                #print("took:%.2fs\t psnr:%.2f name:%s"%(duration_t, p, save_path))
            #print("AVG_DURATION:%.2f\tAVG_PSNR:%.2f"%(total_time/total_imgs, total_psnr/total_imgs))
            avg_psnr = total_psnr/total_imgs
            avg_duration = (total_time/total_imgs)
            if avg_psnr > max[0]:
                max[0] = avg_psnr
                max[1] = epoch

            summary = sess.run(merged, {re_psnr:avg_psnr})
            file_writer.add_summary(summary, epoch)
            tf.logging.warning("AVG_DURATION:%.2f\tAVG_PSNR:%.4f\tepoch:%d"%(avg_duration, avg_psnr, epoch))

        #QP = os.path.basename(ORIGINAL_PATH)
        tf.logging.warning("epoch: %d\tavg_max:%.4f"%(max[1], max[0]))

if __name__ == '__main__':
    test_all_ckpt(MODEL_PATH, [ORIGINAL_PATH, GT_PATH])
