import os
import sys
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='path to dataset')
parser.add_argument('--output', required=True, help='path to output lmdb')

opt = parser.parse_args()
dataset_dir = opt.dataset
output_dir = opt.output

print('dataset_dir', dataset_dir)
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
    # label_file = os.path.join(dataset_dir, 'labels.json')
    #
    # with open(label_file, 'r') as file:
    #     json_data = json.load(file)
    #
    # for key, value in json_data.items():
    #     image_path = os.path.join(dataset_dir, key)
    #     label = value
    #
    #     imagePathList.append(image_path)
    #     labelList.append(label)

    # Create IAM data
    img_dir_2 = '/Users/vng/PycharmProjects/ocr_data/IAM_Handwriting_DB/lines'
    label_file_2 = '/Users/vng/PycharmProjects/ocr_data/IAM_Handwriting_DB/ascii/lines.txt'
    with open(label_file_2, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                line_arr = line.strip().split(' ')
                img_name = line_arr[0]
                img_name_parts = img_name.split('-')
                img_path = os.path.join(img_dir_2, img_name_parts[0],
                                        img_name_parts[0] + '-' + img_name_parts[1], img_name + '.png')
                if os.path.isfile(img_path):
                    imagePathList.append(img_path)
                    text = line_arr[8].replace('|', ' ')
                    labelList.append(text)

    # print(imagePathList)
    # print(labelList)
    createDataset(output_dir, imagePathList, labelList)
