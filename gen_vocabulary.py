import os
import json

label_file = os.path.join('/Users/vng/PycharmProjects/crnn/data/cinnamon_ocr/labels.json')

with open(label_file, 'r') as file:
    json_data = json.load(file)

labelList = []
char_set = set()
for key, value in json_data.items():
    char_set = set.union(char_set, set(value))

with open('./data/vocabulary.txt', 'w') as writer:
    for char in sorted(char_set):
        writer.write(char)