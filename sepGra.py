import os, random, cv2
import numpy as np
from PIL import Image

def getWH(yuvfileName):
    w_included , h_included = os.path.splitext(os.path.basename(yuvfileName))[0].split('x')
    w = w_included.split('_')[-1]
    h = h_included.split('_')[0]
    return int(w), int(h)

def getYdata(path, size):
    w= size[0]
    h=size[1]
    #print(w,h)
    Yt = np.zeros([h, w], dtype="uint8", order='C')
    with open(path, 'rb') as fp:
        fp.seek(0, 0)
        Yt = fp.read()
        tem = Image.frombytes('L', [w, h], Yt)

        Yt = np.asarray(tem, dtype='uint8')


    return Yt

def c_getYdata(path):
    return getYdata(path, getWH(path))

def load_file_list(directory):
    list = []
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        list.append(os.path.join(directory,filename))
    return sorted(list)

def get_train_list(lowList, highList):
    assert len(lowList) == len(highList), "low:%d, high:%d"%(len(lowList), len(highList))
    train_list = []
    for i in range(len(lowList)):
        train_list.append([lowList[i], highList[i]])
    return train_list

def makeDirsIfNotExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def getSobelGradient(imgPath):

    img = c_getYdata(imgPath)

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    absSobelx = cv2.convertScaleAbs(sobelx)
    absSobely = cv2.convertScaleAbs(sobely)

    imgG = cv2.addWeighted(absSobelx,0.5,absSobely,0.5,0)
    return imgG

def getRect(image, startPoint, hBlockSize, wBlockSize):

    data_col = []
    for k in range(hBlockSize):
        data_row = []
        for l in range(wBlockSize):
            data_row.append(image[startPoint[0] + k][startPoint[1] + l])
        data_col.append(data_row)
    #The type should be 'uint8' because the data will be writen into file.
    data_col = np.asarray(data_col, dtype='uint8')
    #print("###")
    #print(data_col)

    return data_col

def seprateGradient(low_dir, high_dir, smallOutput_dir, bigOutput_dir, smallGT_dir, bigGT_dir):

    makeDirsIfNotExist(smallOutput_dir)
    makeDirsIfNotExist(bigOutput_dir)
    makeDirsIfNotExist(smallGT_dir)
    makeDirsIfNotExist(bigGT_dir)
    train_list = get_train_list(load_file_list(low_dir), load_file_list(high_dir))
    processed_train_list = []

    blockSize = 64

    file_cnt = 0
    for i in range(len(train_list)):

        w, h = getWH(train_list[i][0])
        #print(train_list[i][0])

        imageSobel = getSobelGradient(train_list[i][0])
        imageLow = c_getYdata(train_list[i][0])
        imageHigh = c_getYdata(train_list[i][1])

        startPoint = [0,0]
        data_startPoint = [0,0]

        numhBlock = h // 64 if h%64==0 else h // 64 + 1
        numwBlock = w // 64 if w%64==0 else w // 64 + 1

        block_cnt = 0
        for k in range(numhBlock):
            for l in range (numwBlock):
                startPoint = [k * 64, l*64]

                hRemain = h - startPoint[0]
                wRemain = w - startPoint[1]
                if hRemain >= 64:
                    hBlock = 64
                else:
                    continue
                    hBlock = hRemain

                if wRemain >= 64:
                    wBlock = 64
                else:
                    continue
                    wBlock = wRemain


                # abandon blocks that are at eages, which are unable to form input_block with inputSize
                #if (startPoint[0] > h - blockSize or startPoint[1] > w - blockSize):
                #    continue

                lowBlock = getRect(imageLow, startPoint, blockSize, blockSize)
                highBlock = getRect(imageHigh, startPoint, blockSize, blockSize)
                sobelBlock = getRect(imageSobel, startPoint, blockSize, blockSize)

                sobelMean = np.mean(sobelBlock)
                #print(sobelMean)

                # if sobelMean <= 15:
                #     continue
                #     with open(os.path.join(smallOutput_dir, '%d_%d_64x64.yuv'%(file_cnt, block_cnt)), 'w') as f:
                #         lowBlock.tofile(f)
                #     with open(os.path.join(smallGT_dir, '%d_%d_64x64.yuv'%(file_cnt, block_cnt)), 'w') as f:
                #         highBlock.tofile(f)
                #     block_cnt += 1

                # if sobelMean > 15:

                #     with open(os.path.join(bigOutput_dir, '%d_%d_64x64.yuv'%(file_cnt, block_cnt)), 'w') as f:
                #         lowBlock.tofile(f)
                #     with open(os.path.join(bigGT_dir, '%d_%d_64x64.yuv'%(file_cnt, block_cnt)), 'w') as f:
                #         highBlock.tofile(f)
                #     block_cnt += 1


                with open(os.path.join(bigOutput_dir, '%d_%d_64x64.yuv'%(file_cnt, block_cnt)), 'w') as f:
                    lowBlock.tofile(f)
                with open(os.path.join(bigGT_dir, '%d_%d_64x64.yuv'%(file_cnt, block_cnt)), 'w') as f:
                    highBlock.tofile(f)
                block_cnt += 1


        file_cnt += 1

            #assert np.shape(low_data_col) == (blockSize, blockSize)
            #data_col = np.reshape(data_col, (-1))
            #data_col = np.reshape(data_col, (inputSize, inputSize, 1))
            #processed_train_list.append([low_data_col, high_data_col])
        #print("num of list: %d"%len(processed_train_list))

    #return processed_train_list


if __name__ == '__main__':
    low_dir = '/home/chenjs/nf/data/train/av1_2k_qp53'
    high_dir = '/home/chenjs/nf/data/train/av1_2k_gt'
    smallOutput_dir = '/home/chenjs/nf/data/train/av1_gradrow/small_low'
    bigOutput_dir = '/home/chenjs/nf/data/train/av1_gradrow/big_low'
    smallGT_dir = '/home/chenjs/nf/data/train/av1_gradroa/small_high'
    bigGT_dir = '/home/chenjs/nf/data/train/av1_gradrow/big_high'


    seprateGradient(low_dir, high_dir, smallOutput_dir, bigOutput_dir, smallGT_dir, bigGT_dir)
