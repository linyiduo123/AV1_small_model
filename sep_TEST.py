import numpy as np
from PIL import Image
import tensorflow as tf
import os, time
from nf_MODEL import model
#from VDSRMODELold import model
from nf_UTILS import *
from sepGra import getSobelGradient

tf.logging.set_verbosity(tf.logging.WARN)

EXP_DATA = "20190531-av1-2k-noGrad"
TESTOUT_PATH = "./testout/%s/"%(EXP_DATA)
# SMALL_MODEL_PATH = ".\\checkpoints\\20190531-av1-2k-sepGrad-small\\"
SMALL_MODEL_PATH = "D:\\lyd\\cjs_train_all\\nf\\checkpoints\\20190601-av1-2k-sepGrad-small"
BIG_10_MODEL_PATH = "D:\\lyd\\cjs_train_all\\nf\\checkpoints\\20190601-av1-2k-sepGrad-big-continue"
BIG_15_MODEL_PATH = "D:\\lyd\\cjs_train_all\\nf\\checkpoints\\20190616-av1-2k-sepGrad15-big"
BIG_20_MODEL_PATH = "D:\\lyd\\cjs_train_all\\nf\\checkpoints\\20190616-av1-2k-sepGrad20-big"
#MODEL_PATH = "/home/chenjs/a4/chenjs/checkpoints_before_chen/in-loop-filter/0611"
ORIGINAL_PATH = "./av1_18_qp53"
GT_PATH = "./gt"
OUT_DATA_PATH = "./outdata/%s/"%(EXP_DATA)
NOFILTER = {'qp22':41.7929, 'qp27':37.6837, 'qp32':33.9330, 'qp37':30.5175}

##Ground truth images dir should be the 2nd component of 'fileOrDir' if 2 components are given.

##cb, cr components are not implemented
def getRect(image, startPoint, hBlockSize, wBlockSize):

    data_col = []
    for k in range(hBlockSize):
        data_row = []
        for l in range(wBlockSize):
            data_row.append(image[startPoint[0] + k][startPoint[1] + l])
        data_col.append(data_row)
    #The type should be 'uint8' because the data will be writen into file.
    data_col = np.asarray(data_col, dtype='float32')
    #print("###")
    #print(data_col)

    return data_col

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

def test_all_ckpt(fileOrDir):
    max = [0, 0]

    #tem = [f for f in os.listdir(modelPath) if 'data' in f]
    #ckptFiles = sorted([r.split('.data')[0] for r in tem])

    re_psnr = tf.placeholder('float32')
    tf.summary.scalar('re_psnr', re_psnr)

    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        #shared_model = tf.make_template('shared_model', model)
        #output_tensor = shared_model(input_tensor)
        #output_tensor, weight = shared_model(input_tensor)
        output_tensor = model(input_tensor)
        output_tensor = tf.multiply(tf.clip_by_value(output_tensor, 0., 1.), 255)

        #merged = tf.summary.merge_all()
        #file_writer = tf.summary.FileWriter(OUT_DATA_PATH, sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())


        original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)

        # for ckpt in ckptFiles:
        #     epoch = int(ckpt.split('_')[1].split('.')[0])
        #     if epoch < 0:
        #         continue

        #     saver.restore(sess,os.path.join(modelPath,ckpt))
        total_psnr, total_time = 0, 0
        total_imgs = len(fileName_list)
        for i in range(total_imgs):
            imgY = np.squeeze(original_ycbcr[i][0])
            #imgCbCr = original_ycbcr[i][1]
            gtY = np.squeeze(gt_y[i]) if gt_y else 0
            #gtY = gtY * 255

            print(fileName_list[i])
            sobelY = getSobelGradient(fileName_list[i])

            start_t = time.time()
            startPoint = [0,0]
            w, h = getWH(fileName_list[i])
            b = 0
            numElement = 0
            squareSum = 0
            blockSize = 64

            numhBlock = h // 64 if h%64==0 else h // 64 + 1
            numwBlock = w // 64 if w%64==0 else w // 64 + 1
            # print(numhBlock, numwBlock)

            for k in range(numhBlock):
                for l in range(numwBlock):
                    startPoint = [k * 64, l*64]

                    hRemain = h - startPoint[0]
                    wRemain = w - startPoint[1]
                    if hRemain >= 64:
                        hBlock = 64
                    else:
                        hBlock = hRemain

                    if wRemain >= 64:
                        wBlock = 64
                    else:
                        wBlock = wRemain


                    inBlock = getRect(imgY, startPoint, hBlock, wBlock)
                    #print(inBlock.shape)
                    #print(startPoint, hBlock, wBlock)
                    inBlock = np.reshape(inBlock, (1, inBlock.shape[0], inBlock.shape[1], 1))
                    gtBlock = getRect(gtY, startPoint, hBlock, wBlock)
                    sobelBlock = getRect(sobelY, startPoint, hBlock, wBlock)

                    sobelMean = np.mean(sobelBlock)
                    if sobelMean <= 10:
                        # saver.restore(sess,os.path.join(SMALL_MODEL_PATH,'20190601-av1-2k-sepGrad-small_509.ckpt'))
                        print("use small model!")
                        saver.restore(sess,os.path.join(SMALL_MODEL_PATH,'20190601-av1-2k-sepGrad-small_509.ckpt'))
                    elif sobelMean <= 15:
                        # saver.restore(sess,os.path.join(BIG_MODEL_PATH,'20190601-av1-2k-sepGrad-big_049.ckpt'))
                        print("use 10 big model!")
                        saver.restore(sess,os.path.join(BIG_10_MODEL_PATH,'20190601-av1-2k-sepGrad-big-continue_049.ckpt'))
                    elif sobelMean <= 20:
                        # saver.restore(sess,os.path.join(BIG_MODEL_PATH,'20190601-av1-2k-sepGrad-big_049.ckpt'))
                        print("use 15 big model!")
                        saver.restore(sess,os.path.join(BIG_15_MODEL_PATH,'20190616-av1-2k-sepGrad15-big_169.ckpt'))
                    else:
                        # saver.restore(sess,os.path.join(BIG_MODEL_PATH,'20190601-av1-2k-sepGrad-big_049.ckpt'))
                        print("use 20 big model!")
                        saver.restore(sess,os.path.join(BIG_20_MODEL_PATH,'20190616-av1-2k-sepGrad20-big_239.ckpt'))

                    
                    #print(inBlock)
                    out = sess.run(output_tensor, feed_dict={input_tensor: inBlock})
                    out = np.squeeze(out)
                    #print(out)
                    numElement += np.size(gtBlock)
                    squareSum += np.sum(np.square(out - gtBlock))

            assert numElement == w * h, "numElement:%d"%numElement
            p = 10.0 * np.log10(np.square(255.0) / (squareSum / numElement))
            total_psnr += p
            print("filename:%s, psnr:%.4f"%(fileName_list[i], p))

            total_time += time.time() - start_t
            ## gt_y is not empty means 'ground truth' is offered
            # if gt_y:
            #     p = psnr(out, gtY)
            #     total_psnr += p
            #     print("filename:%s, psnr:%.4f"%(fileName_list[i], p))
        #print("AVG_DURATION:%.2f\tAVG_PSNR:%.2f"%(total_time/total_imgs, total_psnr/total_imgs))
        avg_psnr = total_psnr/total_imgs
        avg_duration = (total_time/total_imgs)
        # if avg_psnr > max[0]:
        #     max[0] = avg_psnr
        #     max[1] = epoch

        #summary = sess.run(merged, {re_psnr:avg_psnr})
        #file_writer.add_summary(summary, epoch)
        tf.logging.warning("AVG_DURATION:%.2f\tAVG_PSNR:%.4f\tepoch:%d"%(avg_duration, avg_psnr, 1))

        #QP = os.path.basename(ORIGINAL_PATH)
        #tf.logging.warning("QP:%s\tepoch: %d\tavg_max:%.4f\tdelta:%.4f"%(QP, max[1], max[0], max[0]-NOFILTER[QP]))

if __name__ == '__main__':
    test_all_ckpt([ORIGINAL_PATH, GT_PATH])
