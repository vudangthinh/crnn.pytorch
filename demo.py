import argparse
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
from tool.BeamSearch import ctcBeamSearch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path to pretrain model')
parser.add_argument('--image', required=True, help='path to image file')
opt = parser.parse_args()

model_path = opt.model
img_path = opt.image
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

with open('./data/vocabulary.txt', 'r') as voca_file:
    alphabet = voca_file.readline()

model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((576, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

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
