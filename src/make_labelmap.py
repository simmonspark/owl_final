import json
import yaml
import os


def load_config():
    with open("../config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]
        source_annotations_file = data["annotations_file"]
        num_train_samples = data["num_train_images"]
        num_test_samples = data["num_test_images"]
        img_path = data["images_path"]

    return source_annotations_file, img_path, num_train_samples, num_test_samples


if __name__ == "__main__":
    source_annotations_file, img_dir, num_train_samples, num_test_samples = load_config()
    object_classes = set()
    for dir, _, files in os.walk(source_annotations_file):
        for pick in files:
            fullname = os.path.join(dir, pick)
            with open(fullname, "r") as stream:
                data = json.load(stream)
                img_name = data["images"]["ori_file_name"]
                img_path = os.path.join(dir, img_name)
                img_path = img_path.replace("라벨링데이터", "원천데이터")
                img_path = img_path.replace("TL", "TS")
                for element in data["annotations"]:
                    object_classes.add(element["object_class"])
    object_class_dict = {index: cls for index, cls in enumerate(sorted(object_classes))}
    with open('../data/labelmap.json', "w") as stream:
        json.dump(object_class_dict, stream)