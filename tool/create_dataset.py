import os
import sys
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import argparse
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help='path to dataset')
parser.add_argument('--label', help='path to label file')
parser.add_argument('--output', help='path to output lmdb')

opt = parser.parse_args()
image_dir = opt.image_dir
label_file = opt.label
output_dir = opt.output

print('image_dir', image_dir)
print('label file', label_file)
print('output_dir', output_dir)

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8')
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode('utf-8')
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    imagePathList = []
    labelList = []

    # Create cinnamon data
    # with open(label_file, 'r') as file:
    #     json_data = json.load(file)
    #
    # for key, value in json_data.items():
    #     img_name = key.split('.')[0]
    #     image_path = os.path.join(image_dir, key)
    #     label = value
    #
    #     # imagePathList.append(os.path.join(image_dir, img_name + '_gray.png'))
    #     # imagePathList.append(os.path.join(image_dir, img_name + '_crop_thread_0.png'))
    #     # imagePathList.append(os.path.join(image_dir, img_name + '_thread_10.png'))
    #     imagePathList.append(os.path.join(image_dir, img_name + '_crop_thread_20.png'))
    #     # imagePathList.append(os.path.join(image_dir, img_name + '_crop_thread_otsu.png'))
    #     # labelList.append(label)
    #     # labelList.append(label)
    #     # labelList.append(label)
    #     labelList.append(label)
    #     # labelList.append(label)

    # Create IAM data
    img_dir_2 = '/Users/thinhvu/PycharmProjects/ocr_data/IAM_Handwriting_DB/sentences'
    label_file_2 = '/Users/thinhvu/PycharmProjects/ocr_data/IAM_Handwriting_DB/ascii/sentences.txt'
    with open(label_file_2, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                line_arr = line.strip().split(' ')
                if line_arr[2] == 'ok':
                    graylevel = int(line_arr[3])

                    img_name = line_arr[0]
                    img_name_parts = img_name.split('-')
                    img_path = os.path.join(img_dir_2, img_name_parts[0],
                                            img_name_parts[0] + '-' + img_name_parts[1], img_name + '_binary.png')

                    # save_img_path = os.path.join(img_dir_2, img_name_parts[0],
                    #                         img_name_parts[0] + '-' + img_name_parts[1], img_name + '_binary.png')
                    #
                    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # img = cv2.threshold(img, graylevel, 255, cv2.THRESH_BINARY)[1]
                    # cv2.imwrite(save_img_path, img)

                    if os.path.isfile(img_path):
                        imagePathList.append(img_path)
                        text = line_arr[-1].replace('|', ' ')
                        labelList.append(text)

    # print(imagePathList)
    # print(labelList)
    createDataset(output_dir, imagePathList, labelList)
