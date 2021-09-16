import os
from scipy import io
import numpy as np
import cv2
import glob
import itertools

BALL_LABEL = 0
PLAYER_LABEL = 1


def merge_ball_annotation(ball_anno_path, dataset_path):
    assert os.path.exists(ball_anno_path), 'Cannot find ball annotations: ' + str(ball_anno_path)

    labels_path = os.path.join(dataset_path, 'labels')
    assert os.path.exists(labels_path), 'Cannot find labels: ' + str(labels_path)

    ball_count = 0
    for file1, file2 in itertools.product(
                    sorted(glob.glob(labels_path+'/*.txt*')),sorted(glob.glob(ball_anno_path+'/*.txt*'))):
        if file1.split('/')[-1].split('.')[0] == file2.split('/')[-1].split('.')[0]:
            with open(file1, 'a') as append_file, open(file2, 'r') as read_file:
                for line in read_file.readlines():
                    if line != '\n':
                        line = line.split()
                        newline = str(BALL_LABEL)+' '+line[1]+' '+line[2]+' '+line[3]+' '+line[4]
                        append_file.write(newline)
                        append_file.write('\n')

                        ball_count += 1

    print("{} balls in SPD DataSet".format(ball_count))


def open_dataset(root, ndx):
    # Get path to images and ground truth
    assert os.path.exists(root), print('Dataset root not found: {}'.format(root))
    assert ndx in [1, 2], print('Dataset index can be only 1 or 2')

    gt_path = os.path.join(root, 'annotation_{}.mat'.format(ndx))
    assert os.path.exists(gt_path), print('Ground truth not found: {}'.format(gt_path))

    gt = io.loadmat(gt_path)
    return gt


def get_annotations(bboxes):
    # creating a array with PLAYER_LABEL
    labels = np.full((bboxes.shape[0], 1), PLAYER_LABEL, dtype=int)

    labels = np.column_stack((labels, bboxes))

    return labels


def spd_format_to_yolo_format(xtl, ytl, xbr, ybr, frame):
    # return bounding box co_ordinates in (norm_x_center, norm_y_center, norm_width, norm_height) format
    img_h, img_w = frame[0], frame[1]

    w = xbr - xtl
    h = ybr - ytl
    xc = xtl + w/2
    yc = ytl + h/2

    norm_xc, norm_yc, norm_w, norm_h = xc/img_w, yc/img_h, w/img_w, h/img_h
    return norm_xc, norm_yc, norm_w, norm_h


def create_spd_dataset(dataset_path, mode):
    assert mode == 'train' or mode == 'val'
    if mode == 'train':
        mode_id = 1
        ground_truth = open_dataset(dataset_path, mode_id)
    else:
        mode_id = 2
        ground_truth = open_dataset(dataset_path, mode_id)

    ground_truth = ground_truth['annot'][0]

    frame_count = 0
    players_count= 0
    for bboxes, img_name in ground_truth:
        if mode == 'train':
            img_name = 'DataSet_001_'+img_name[0]
        else:
            img_name = 'DataSet_002_'+img_name[0]

        # Verify if the image file exists, if not, skip
        image_path = dataset_path+'/images/'+img_name
        if not os.path.exists(image_path):
            continue
        frame = cv2.imread(image_path)
        labels = get_annotations(bboxes)
        labels_dir = dataset_path+'/labels'
        if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)
        labels_path = labels_dir+'/'+img_name.replace(".jpg", ".txt")

        with open(labels_path, "w") as fp:
            for row in labels:
                label, xtl, ytl, xbr, ybr = row[0], row[1], row[2], row[3], row[4]
                norm_xc, norm_yc, n_w, n_h = spd_format_to_yolo_format(xtl, ytl, xbr, ybr, frame.shape)
                line = str(label)+' '+str(norm_xc)+' '+str(norm_yc)+' '+str(n_w)+' '+str(n_h)
                fp.write(line)
                fp.write('\n')

                players_count += 1
        frame_count += 1

    print("{} frames, {} players in DataSet_{}".format(players_count, frame_count, '001' if mode=='train' else '002'))


if __name__ == '__main__':
    dataset_path = '/home/sysadm/Desktop/soccer_analytics/dataset/SPD'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    # renaming all images in DataSet_001 and DataSet_002 and moving to images folder
    images_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    for dir in ['DataSet_001', 'DataSet_002']:
        collection = os.path.join(dataset_path, dir)

        for i, filename in enumerate(os.listdir(collection)):
            os.rename(collection+'/'+filename, images_path+'/'+filename)
    

    for mode in ['train', 'val']:
        label_path = create_spd_dataset(dataset_path, mode)

    ball_anno_path = '/home/sysadm/Desktop/soccer_analytics/dataset/SPD/ball_annotation(YOLO format)'
    merge_ball_annotation(ball_anno_path, dataset_path)
