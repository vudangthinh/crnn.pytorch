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

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path to pretrain model')
parser.add_argument('--image', help='path to image file')
parser.add_argument('--trainRoot', default='./data', help='path to dataset')
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

if img_path:
    transformer = dataset.resizeNormalize((576, 32))
    image = Image.open(img_path).convert('L')
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
    print('%-20s => %-20s' % (raw_pred, sim_pred))

else:
    train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
    assert train_dataset

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = 64
    batch_size = split
    valid_idx = indices[:split]

    valid_sampler = SubsetRandomSampler(valid_idx)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=split,
        shuffle=False, sampler=valid_sampler,
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
    for i_batch, (cpu_images, cpu_texts) in enumerate(valid_loader):
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
            loss_dict[gt] = loss
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    sorted_by_value = sorted(loss_dict.items(), key=lambda kv: kv[1])
    for item in sorted_by_value:
        print(item)