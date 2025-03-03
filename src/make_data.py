import json
import random
from collections import defaultdict, Counter, OrderedDict
import yaml
import os

with open('../data/labelmap.json', 'r') as f:
    labelmap = json.load(f)
inverse = dict()
for k, v in labelmap.items():
    inverse[v] = k


def load_config():
    with open("../config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]
        source_annotations_file = data["annotations_file"]
        num_train_samples = data["num_train_images"]
        num_test_samples = data["num_test_images"]
        img_path = data['images_path']

    return source_annotations_file, img_path, num_train_samples, num_test_samples


def shuffle_indices(subset_indices, num_train_samples, num_test_samples):
    random.shuffle(subset_indices)
    train_indices = subset_indices[:num_train_samples]
    test_indices = subset_indices[
                   num_train_samples: num_train_samples + num_test_samples
                   ]
    return train_indices, test_indices


if __name__ == "__main__":
    source_annotations_file, img_dir, num_train_samples, num_test_samples = load_config()
    store = []
    for dir, _, file in os.walk(source_annotations_file):
        for pick in file:
            fullname = os.path.join(dir, pick)
            with open(fullname, "r") as stream:
                data = json.load(stream)
                img_name = data['images']['ori_file_name']
                img_path = os.path.join(dir, img_name)
                img_path = img_path.replace('라벨링데이터', '원천데이터')
                img_path = img_path.replace('TL', 'TS')
                if os.path.exists(img_path) is False: continue
                tmp = []
                if len(data['annotations'])<=1: continue
                for element in data['annotations']:
                    query = element["object_class"]
                    cls = int(inverse[query])
                    bbox=[]
                    bbox.extend(element['bbox'][0])
                    bbox.extend(element['bbox'][1])
                    tmp.append({'label' : cls, 'bbox' : bbox})
                store.append({img_path: tmp})
    store = random.sample(store, len(store))
    size = len(store)
    print(size)
    train = store[:int(size * 0.98)]
    test = store[int(size * 0.98):]


    with open('../data/train.json', 'w', encoding='utf-8') as stream:
        json.dump(train, stream)
        print('train.json created.\n')

    with open('../data/test.json', 'w', encoding='utf-8') as stream:
        json.dump(test, stream)
        print('test.json created.\n')
