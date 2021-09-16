import os
import glob
import itertools
import numpy as np

def rename_images_labels(dataset_path):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    count = 0
    for dataset in ['ISSIA', 'SPD', 'MOUNSIF']:

        image_path = os.path.join(dataset_path, dataset, 'images')
        label_path = os.path.join(dataset_path, dataset, 'labels')

        for image, label in itertools.product(
                        sorted(glob.glob(image_path+'/*.jpg*')),sorted(glob.glob(label_path+'/*.txt*'))):
            if image.split('/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0]:
                count += 1

                image_name = str(count)+'.jpg'
                label_name = str(count)+'.txt'

                os.rename(image, os.path.join(images_dir, image_name))
                os.rename(label, os.path.join(labels_dir, label_name))

def split_train_val(dataset_path, val_ratio):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    assert os.path.exists(images_dir)
    assert os.path.exists(labels_dir)

    all_images_names = os.listdir(images_dir)
    np.random.shuffle(all_images_names)
    train_img_names, val_img_names = np.split(np.array(all_images_names),[int(len(all_images_names)* (1 - val_ratio))])

    with open(dataset_path+'/train.txt', 'a') as append_file:
        for img in train_img_names:
            line = 'data/obj/'+str(img)
            append_file.write(line)
            append_file.write('\n')

    with open(dataset_path+'/val.txt', 'a') as append_file:
        for img in val_img_names:
            line = 'data/obj/'+str(img)
            append_file.write(line)
            append_file.write('\n')



if __name__ == '__main__':

    dataset_path = '/home/sysadm/Desktop/soccer_analytics/dataset'
    rename_images_labels(dataset_path)
    split_train_val(dataset_path, val_ratio=0.3)
