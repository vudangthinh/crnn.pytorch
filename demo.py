import argparse
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
from tool.BeamSearch import ctcBeamSearch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import operator
import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path to pretrain model')
parser.add_argument('--image', help='path to image file')
parser.add_argument('--image_dir', help='path to image directory')
parser.add_argument('--label_file', help='path to label file')
parser.add_argument('--trainRoot', default='./data', help='path to dataset')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')

opt = parser.parse_args()

model_path = opt.model
img_path = opt.image
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

with open('./data/vocabulary.txt', 'r') as voca_file:
    alphabet = voca_file.readline()

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    print('device:', torch.cuda.current_device())

crnn = crnn.CRNN(32, 1, len(alphabet) + 1, 256)
if torch.cuda.is_available():
    crnn = crnn.cuda()
print('loading pretrained model from %s' % model_path)
crnn.load_state_dict(torch.load(model_path))
crnn.eval()

converter = utils.strLabelConverter(alphabet)

def test_image(image_path, label, keep_ratio=False):
    image = Image.open(image_path).convert('L')
    if keep_ratio:
        h, w = image.shape
        resize_w = 32.0 * w / h
        transformer = dataset.resizeNormalize((resize_w, 32))
    else:
        transformer = dataset.resizeNormalize((576, 32))

    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    preds = crnn(image)

    # preds_soft = F.softmax(preds, dim=2)
    # print("alphabet:", len(alphabet))
    # print(preds_soft.shape)
    # preds_beam = preds_soft.squeeze()
    # print(preds_beam.shape)
    # beam_pred = ctcBeamSearch(preds_beam, alphabet, None)
    # print('Beam Pred:', beam_pred)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    loss = utils.cer_loss_one_image(sim_pred, label)
    print('%-20s => %-20s - loss: %f' % (raw_pred, sim_pred, loss))

    return loss

if img_path:
    test_image(img_path, '', opt.keep_ratio)

elif opt.image_dir:
    with open(opt.label_file, 'r') as f:
        labels = json.load(f)

    total_loss = 0
    for key, value in labels.items():

        image_path = os.path.join(opt.image_dir, key.split('.')[0] + '_crop_thread_20.png')
        print(image_path)
        loss = test_image(img_path, value, opt.keep_ratio)
        total_loss += loss

    print("CER Loss: ", total_loss * 1.0 / len(labels))

elif opt.trainRoot:
    train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
    assert train_dataset

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = 64
    batch_size = split
    valid_idx = indices[:split]

    # valid_sampler = SubsetRandomSampler(valid_idx)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, sampler=None,
        num_workers=int(4),
        collate_fn=dataset.alignCollate(imgH=32, imgW=576))

    image = torch.FloatTensor(batch_size, 3, 32, 576)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)

    if torch.cuda.is_available():
        crnn = crnn.cuda(opt.gpu)
        image = image.cuda(opt.gpu)

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    loss_dict = {}
    total_loss = 0
    for i_batch, (cpu_images, cpu_texts) in enumerate(valid_loader):
        print('batch: ', i_batch)
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            loss = utils.cer_loss_one_image(pred, gt)
            total_loss += loss
            loss_dict[gt] = loss
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    sorted_by_value = sorted(loss_dict.items(), key=lambda kv: kv[1])
    for item in sorted_by_value:
        print(item)

    print('Valid Cer Loss:', total_loss/len(train_dataset))


