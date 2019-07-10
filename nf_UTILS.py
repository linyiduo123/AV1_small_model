import numpy as np
import tensorflow as tf
import math, os, random, time
from scipy.linalg import hadamard
import scipy.misc
from PIL import Image
from nf_UNK import BATCH_SIZE
from nf_UNK import PATCH_SIZE
from nf_UNK import IMAGE_BATCH

def normalize(x):
    x = x / 255.
    return truncate(x, 0., 1.)

def denormalize(x):
    x = x * 255.
    return truncate(x, 0., 255.)

def truncate(input, min, max):
    input = np.where(input > min, input, min)
    input = np.where(input < max, input, max)
    return input

def remap(input):
    input = 16+219/255*input
    #return tf.clip_by_value(input, 16, 235).eval()
    return truncate(input, 16.0, 235.0)

def deremap(input):
    input = (input-16)*255/219
    #return tf.clip_by_value(input, 0, 255).eval()
    return truncate(input, 0.0, 255.0)

def makeDirsIfNotExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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

# def prepare_nn_data(train_list, idx_img=None):
#     i = np.random.randint(len(train_list)) if (idx_img is None) else idx_img
#     input_image  = c_getYdata(train_list[i][0])
#     gt_image = c_getYdata(train_list[i][1])


#     input_list = []
#     gt_list = []
#     inputcbcr_list = []

#     for idx in range(BATCH_SIZE):

#         #crop images to the disired size.
#         input_imgY, gt_imgY = crop(input_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")

#         input_imgY = Greying(input_imgY, PATCH_SIZE[0])
#         #scipy.misc.imsave('/home/chenjs/relHM/tFromPython/%s.png'%(time.time()), input_imgY)
#         #scipy.misc.imsave('/home/chenjs/relHM/tFromPython/%s.png'%(time.time()), gt_imgY)

#         #normalize
#         input_imgY = normalize(input_imgY)
#         gt_imgY = normalize(gt_imgY)

#         input_list.append(input_imgY)
#         gt_list.append(gt_imgY)
#         #inputcbcr_list.append(input_imgCbCr)

#     input_list = np.reshape(input_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
#     gt_list = np.reshape(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
#     #inputcbcr_list = np.resize(inputcbcr_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 2))

#     return input_list, gt_list, inputcbcr_list

def prepare_nn_data(train_list, idx_img=None):
    batchSizeRandomList = random.sample(range(0, len(train_list)), IMAGE_BATCH)
    #print(batchSizeRandomList)
    gt_list = []
    lowData_list = []
    inputcbcr_list = []
    for i in batchSizeRandomList:
        lowData_image = c_getYdata(train_list[i][0])
        #print(train_list[i][0])
        gt_image = c_getYdata(train_list[i][1])
        for j in range(BATCH_SIZE):
            # crop images to the disired size.
            lowData_imgY, gt_imgY = crop(lowData_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")

            # normalize
            lowData_imgY = normalize(lowData_imgY)
            gt_imgY = normalize(gt_imgY)

            lowData_list.append(lowData_imgY)
            gt_list.append(gt_imgY)

    lowData_list = np.reshape(lowData_list, (BATCH_SIZE * IMAGE_BATCH, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.reshape(gt_list, (BATCH_SIZE * IMAGE_BATCH, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return lowData_list, gt_list, inputcbcr_list

def getLdata(path):
    img = Image.open(path)
    return  np.asarray(img, dtype='uint8')

def Greying(block, patch_size):
    newBlock = [[0 for i in range(patch_size)] for j in range(patch_size)]
    for i in range(patch_size):
        for j in range(patch_size):
            if (i >= patch_size // 2 and j >= patch_size // 2):
                newBlock[i][j] = 128
                continue
            newBlock[i][j] = block[i][j]
    return np.asarray(newBlock)

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

        Yt = np.asarray(tem, dtype='float32')

        # for n in range(h):
        #     for m in range(w):
        #         Yt[n, m] = ord(fp.read(1))

    return Yt

def c_getYdata(path):
    return getYdata(path, getWH(path))

def calcHADLoss(ori, cur):
    ori = tf.squeeze(ori)
    cur = tf.squeeze(cur)
    resi = tf.subtract(ori, cur)

    transfMatrix = hadamard(8, dtype='float32')

    assert resi.shape[0] == BATCH_SIZE
    resiList = [resi[a] for a in range(BATCH_SIZE)]

    totalSATD = 0
    for i in range(BATCH_SIZE):
        resiPatch = tf.reshape(resiList[i], [1024])

        for col in range(4):
            for row in range(4):
                start = col*8*PATCH_SIZE[0]+row*8
                qt = [[0 for i in range(8)] for j in range(8)]
                for i in range(8):
                    for j in range(8):
                        qt[i][j] = resiPatch[start + i * PATCH_SIZE[0] + j]
                transfromed = tf.matmul(transfMatrix, qt)
                transfromed = tf.matmul(transfromed, transfMatrix)

                totalSATD += tf.reduce_sum(tf.abs(transfromed))

    return totalSATD

def img2y(input_img):
    if np.asarray(input_img).shape[2] == 3:
        input_imgY = input_img.convert('YCbCr').split()[0]
        input_imgCb, input_imgCr = input_img.convert('YCbCr').split()[1:3]

        input_imgY = np.asarray(input_imgY, dtype='float32')
        input_imgCb = np.asarray(input_imgCb, dtype='float32')
        input_imgCr = np.asarray(input_imgCr, dtype='float32')


        #Concatenate Cb, Cr components for easy, they are used in pair anyway.
        input_imgCb = np.expand_dims(input_imgCb,2)
        input_imgCr = np.expand_dims(input_imgCr,2)
        input_imgCbCr = np.concatenate((input_imgCb, input_imgCr), axis=2)

    elif np.asarray(input_img).shape[2] == 1:
        print("This image has one channal only.")
        #If the num of channal is 1, remain.
        input_imgY = input_img
        input_imgCbCr = None
    else:
        print("The num of channal is neither 3 nor 1.")
        exit()
    return input_imgY, input_imgCbCr

def crop(input_image, gt_image, patch_width, patch_height, img_type):
    assert type(input_image) == type(gt_image), "types are different."
    #return a ndarray object
    if img_type == "ndarray":
        in_row_ind   = random.randint(0,input_image.shape[0]-patch_width)
        in_col_ind   = random.randint(0,input_image.shape[1]-patch_height)

        input_cropped = input_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]
        gt_cropped = gt_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]

    #return an "Image" object
    elif img_type == "Image":
        in_row_ind   = random.randint(0,input_image.size[0]-patch_width)
        in_col_ind   = random.randint(0,input_image.size[1]-patch_height)

        input_cropped = input_image.crop(box=(in_row_ind, in_col_ind, in_row_ind+patch_width, in_col_ind+patch_height))
        gt_cropped = gt_image.crop(box=(in_row_ind, in_col_ind, in_row_ind+patch_width, in_col_ind+patch_height))

    return input_cropped, gt_cropped

def save_images(inputY, inputCbCr, size, image_path):
    """Save mutiple images into one single image.

    Parameters
    -----------
    images : numpy array [batch, w, h, c]
    size : list of two int, row and column number.
        number of images should be equal or less than size[0] * size[1]
    image_path : string.

    Examples
    ---------
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> tl.visualize.save_images(images, [8, 8], 'temp.png')
    """
    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img

    inputY = inputY.astype('uint8')
    inputCbCr = inputCbCr.astype('uint8')
    output_concat = np.concatenate((inputY, inputCbCr), axis=3)

    assert len(output_concat) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(output_concat))

    new_output = merge(output_concat, size)

    new_output = new_output.astype('uint8')

    img = Image.fromarray(new_output, mode='YCbCr')
    img = img.convert('RGB')
    img.save(image_path)

def get_image_batch(train_list,offset,batch_size):
    target_list = train_list[offset:offset+batch_size]
    input_list = []
    gt_list = []
    inputcbcr_list = []
    for pair in target_list:
        input_img = Image.open(pair[0])
        gt_img = Image.open(pair[1])

        #crop images to the disired size.
        input_img, gt_img = crop(input_img, gt_img, PATCH_SIZE[0], PATCH_SIZE[1], "Image")

        #focus on Y channal only
        input_imgY, input_imgCbCr = img2y(input_img)
        gt_imgY, gt_imgCbCr = img2y(gt_img)

        #input_imgY = normalize(input_imgY)
        #gt_imgY = normalize(gt_imgY)

        input_list.append(input_imgY)
        gt_list.append(gt_imgY)
        inputcbcr_list.append(input_imgCbCr)

    input_list = np.resize(input_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.resize(gt_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return input_list, gt_list, inputcbcr_list

def save_test_img(inputY, inputCbCr, path):
    assert len(inputY.shape) == 4, "the tensor Y's shape is %s"%inputY.shape
    assert inputY.shape[0] == 1, "the fitst component must be 1, has not been completed otherwise.{}".format(inputY.shape)

    inputY = np.squeeze(inputY, axis=0)
    inputY = inputY.astype('uint8')

    inputCbCr = inputCbCr.astype('uint8')

    output_concat = np.concatenate((inputY, inputCbCr), axis=2)
    img = Image.fromarray(output_concat, mode='YCbCr')
    img = img.convert('RGB')
    img.save(path)

def psnr(hr_image, sr_image, max_value=255.0):
    eps = 1e-10
    if((type(hr_image)==type(np.array([]))) or (type(hr_image)==type([]))):
        hr_image_data = np.asarray(hr_image, 'float32')
        sr_image_data = np.asarray(sr_image, 'float32')

        diff = sr_image_data - hr_image_data
        mse = np.mean(diff*diff)
        mse = np.maximum(eps, mse)
        return float(10*math.log10(max_value*max_value/mse))
    else:
        assert len(hr_image.shape)==4 and len(sr_image.shape)==4
        diff = hr_image - sr_image
        mse = tf.reduce_mean(tf.square(diff))
        mse = tf.maximum(mse, eps)
        return 10*tf.log(max_value*max_value/mse)/math.log(10)


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess, val_set, images_pl, labels_pl, logits, epoch=-1):
    batches = len(val_set) // BATCH_SIZE
    num_example = batches * BATCH_SIZE
    num_correct = 0
    eval_correct = evaluation(logits, labels_pl)
    for i in range(batches):
        val_images, val_labels = get_batch_data(val_set, BATCH_SIZE, i)
        this_correct, out = sess.run([eval_correct, logits], {images_pl:val_images, labels_pl:val_labels})
        num_correct += this_correct
        print(np.squeeze(val_images))
        print("####################################################")
        print(out)
    percision = num_correct * 1.0 / num_example
    tf.logging.warning("Epoch:%d, num correct:%d, num example:%d, percision:%04f"%(epoch, num_correct, num_example, percision))

# def get_train_pair(yuv_dir, txt_dir, devideSize, inputSize, set_type):
#     '''
#     Args
#     devideSize: CU size

#     inputSize: expected block size of the input of CNN. Generally devideSize is samll, is need to be expanded.

#     set_type: possible value [train, valid, test]. set type determines the number of a set, and whether select randomly.


#     Return
#     A set of [imageData, label]. imagedata is a vevtor with size of inputSize*inputSize, which is expanded from original block with size of devideSize*devideSize
#     '''


#     #Get pairs of [imageFilename, labelFilename]
#     train_list = get_train_list(load_file_list(yuv_dir), load_file_list(txt_dir))
#     processed_train_list = []

#     #Using margin to determine if the expanded block of a original block exists
#     assert (inputSize - devideSize) % 2 == 0
#     margin = (inputSize - devideSize) // 2


#     for i in range(len(train_list)):
#         w, h = getWH(train_list[i][0])
#         print(train_list[i][0])
#         image = c_getYdata(train_list[i][0])
#         label = np.loadtxt(train_list[i][1], dtype='uint8').reshape(-1)
#         assert len(label) == (w * h) / (devideSize * devideSize), "label:%d, (w*h)/(devideSize^2)"%(len(label), (w * h) / (devideSize * devideSize))

#         startPoint = [0,0]
#         data_startPoint = [0,0]
#         for j in range(len(label)):

#             startPoint = [((j*devideSize) // w) * devideSize, (j*devideSize) % w] #[col, row]

#             # abandon blocks that are at eages, which are unable to form input_block with inputSize
#             if (startPoint[0] < margin or startPoint[0] > (h - margin - devideSize) or startPoint[1] < margin or startPoint[1] > (w - margin - devideSize)):
#                 continue

#             if (set_type == 'train'):
#                 # randomly skip the current block, and roughly select 1/10 blocks of one image
#                 if (random.randrange(10) is not 3):
#                     continue
#                 if (set_type == 'train'):
#                     # if 10000 blocks have been selected, skip to the next image
#                     if (len(processed_train_list) // 1000 > i):
#                         break

#             data_startPoint = [startPoint[0]-margin, startPoint[1]-margin]
#             data_col = []
#             for k in range(inputSize):
#                 data_row = []
#                 for l in range(inputSize):
#                     data_row.append(image[data_startPoint[0] + k][data_startPoint[1] + l])
#                 data_col.append(data_row)
#             assert np.shape(data_col) == (inputSize, inputSize)
#             data_col = np.reshape(data_col, (-1))
#             #data_col = np.reshape(data_col, (inputSize, inputSize, 1))
#             processed_train_list.append([data_col, label[j]])
#         print("num of list: %d"%len(processed_train_list))

#     return processed_train_list

def get_train_pair(data_dir, label_dir):
    #Get pairs of [imageFilename, labelFilename]
    train_list = get_train_list(load_file_list(data_dir), load_file_list(label_dir))
    processed_train_list = []

    for i in range(len(train_list)):

        #if (getWH(train_list[i][0]) != (PATCH_SIZE, PATCH_SIZE)):
        #    continue

        image = c_getYdata(train_list[i][0])
        label = c_getYdata(train_list[i][1])
        #w, h  = getWH(train_list[i][0])

        image = normalize(image)

        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        label = np.reshape(label, (label.shape[0], label.shape[1], 1))

        #assert image.shape[0] == 64 and image.shape[1] == 64, "%d, %d"%(image.shape[0], image.shape[1])
        #assert label.shape[0] == 64 and label.shape[1] == 64


        processed_train_list.append([image, label])

    return processed_train_list


def get_batch_data(train_list, batch_size, idx):
    images = []
    labels =[]

    for i in range(batch_size):
        images.append(train_list[idx*batch_size + i][0])
        labels.append(train_list[idx*batch_size + i][1])
    return images, labels

