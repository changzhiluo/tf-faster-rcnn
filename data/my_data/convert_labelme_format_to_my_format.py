# -*- coding:utf-8 -*-
#!/usr/bin/env python

import json
import os.path

pre_path = './xuebi/' # prefix of "file_name" element of the output json file
rootdir = './xuebi'  # folder path of labeled image 

categories = [
    {
      "name": "xuebi",
      "id": 1,
      "supercategory": "drink"
    }
  ]

licenses = [
    {
      "name": "woshishabi",
      "id": 1,
      "url": "None"
    }
  ]

info = {
    "contributor": "Changzhi Luo",
    "version": "1.0",
    "description": "CUB-200-2011 Dataset",
    "url": "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
    "date_created": "2018/11/28",
    "year": "2008"
  }

images = []

annotations = []


def get_file_name(rootdir):
    file_list = []
    tmp_list = os.listdir(rootdir)
    for i in range(0, len(tmp_list)):
        path = os.path.join(rootdir, tmp_list[i])
        if os.path.isfile(path):
            if os.path.splitext(path)[-1][1:] == 'json':
                file_list.append(path)
    return file_list


def get_json_elements(file_list):
    global images, annotations, pre_path
    a_nb = 0
    for i in range(len(file_list)):
        # print i
        image = {
            'id': 0,
            'width': 0,
            'height': 0,
            'file_name': ''}

        annotation = {
            'id': 0,
            'bbox': [0, 0, 0, 0],
            'image_id': 0,
            'category_id': 0,
        }

        with open(file_list[i], "r") as f:
            temp = json.loads(f.read())

            image['id'] = i
            image['width'] = temp['imageWidth']
            image['height'] = temp['imageHeight']
            image['file_name'] = pre_path + temp['imagePath']
            images.append(image)

            for j in range(len(temp['shapes'])):
                annotation['id'] = a_nb
                a_nb = a_nb + 1
                annotation['bbox'] = [temp['shapes'][j]['points'][0][0],
                                      temp['shapes'][j]['points'][0][1],
                                      temp['shapes'][j]['points'][1][0],
                                      temp['shapes'][j]['points'][1][1]]
                annotation['image_id'] = i
                annotation['category_id'] = int(temp['shapes'][j]['label'])
                annotations.append(annotation)
            f.close()

        # print images, annotations


def output():
    global images
    print(json.dumps(images))
    with open('new_data.json', 'w') as json_file:
        json_file.write(json.dumps(
            {'categories': categories,
             'images': images,
             'annotations': annotations,
             'licenses': licenses,
             'info': info},
            indent=2, ensure_ascii=False))


def main():
    global rootdir
    file_list = get_file_name(rootdir)
    get_json_elements(file_list)
    output()


if __name__ == '__main__':
    main()

