import os
import subprocess
from xml.dom import minidom
from collections import defaultdict
import numpy as np
import cv2
import glob

# Labels starting from 0
BALL_LABEL = 0
PLAYER_LABEL = 1
# Size of the ball bbox in pixels (fixed as we detect only ball center)
BALL_BBOX_SIZE = 20
# Dictionary with ground truth annotations per camera
gt_annotations = {}

class SequenceAnnotations:
    '''
    Class for storing annotations for the video sequence
    '''
    def __init__(self):
        # ball_pos contains list of ball positions (x,y) on each frame; multiple balls per frame are possible
        self.ball_pos = defaultdict(list)
        # persons contains a list of bounding boxes for players visible on the frame
        self.persons = defaultdict(list)

def extract_frames(dataset_path, camera_id, image_path):
    # Extract frames from the sequence
    query = "ffmpeg -i "+dataset_path+"/Film\ Role-0\ ID-"+str(camera_id)+"\ T-2\ m00s00-000-m00s00-185.avi -qscale:v 2 -crf 18 "+image_path+"/"+str(camera_id)+"%4d.jpg -loglevel quiet"
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()


def parse_framespan(framespan):
    '''
    Auxiliary function to parse frame span value
    :param framespan: string in "start_frame:end_frame" format, e.g. "1182:1185"
    :return: a tuple of int (start_frame, end_frame)
    '''
    # Parse framespan, format is: framespan="1182:1185"
    pos = framespan.find(':')
    if pos < 1:
        assert False, 'Incorrect framespan value: ' + framespan

    start_frame = int(framespan[:pos])
    end_frame = int(framespan[pos + 1:])
    return start_frame, end_frame


def load_groundtruth(filepath):
    '''
    Load ground truth data from XML file (ISSIA dataset)
    :param filepath: Path to the ground truth XML file
    :return: Dictionary with ISSIA ground truth data. The dictionary has the following elements:
             gt['BallPos']              - ball positions at each frame
             gt['Person']               - players bounding boxes
    '''

    assert os.path.isfile(filepath)
    xmldoc = minidom.parse(filepath)

    # Processs information from file section. Extract number of frames (NUMFRAMES) value.
    itemlist = xmldoc.getElementsByTagName('file')
    # There should be exactly 1 file element in a groundtruth file
    assert len(itemlist) == 1

    num_frames = None

    for e in itemlist[0].getElementsByTagName('attribute'):
        if e.attributes['name'].value =='NUMFRAMES':
            values = e.getElementsByTagName('data:dvalue')
            # There should be only one data:dvalue node
            assert len(values) == 1
            num_frames = values[0].attributes['value'].value
            break

    if num_frames is None:
        assert False, 'NUMFRAMES not definedin XML file.'

    # Processs information from data section. Extract ball positions
    # Dictionary to hold ground truth values
    gt ={}
    # List of ball position on each frame from the sequence
    gt['BallPos'] = []
    # Dictionary storing list of bounding boxes for each frame from the sequence
    gt['Person'] = []

    itemlist = xmldoc.getElementsByTagName('object')
    # This returns multiple object elements (BALL, Person)

    for e in itemlist:
        assert 'name' in e.attributes
        if e.attributes['name'].value == 'BALL':
            ball_attributes = e.getElementsByTagName('attribute')
            # Valid ball attributes are: BallPos, BallShot, PlayerInteractingID

            for elem in ball_attributes:
                if elem.attributes['name'].value == 'BallPos':
                    for e in elem.getElementsByTagName('data:point'):
                        # <data:point framespan='1182:1182' x='91' y='443'/>
                        assert 'framespan' in e.attributes
                        assert 'x' in e.attributes
                        assert 'y' in e.attributes

                        framespan = e.attributes['framespan'].value
                        x = int(e.attributes['x'].value)
                        y = int(e.attributes['y'].value)
                        start_frame, end_frame = parse_framespan(framespan)
                        gt['BallPos'].append((start_frame, end_frame, x, y))

        elif e.attributes['name'].value == 'Person':
            person_id = e.attributes['id'].value
            person_attributes = e.getElementsByTagName('attribute')
            for elem in person_attributes:
                if elem.attributes['name'].value == 'LOCATION':
                    for e in elem.getElementsByTagName('data:bbox'):
                        # <data:point framespan='1182:1182' x='91' y='443'/>
                        assert 'framespan' in e.attributes
                        assert 'height' in e.attributes
                        assert 'width' in e.attributes
                        assert 'x' in e.attributes
                        assert 'y' in e.attributes

                        framespan = e.attributes['framespan'].value
                        height = int(e.attributes['height'].value)
                        width = int(e.attributes['width'].value)
                        x = int(e.attributes['x'].value)
                        y = int(e.attributes['y'].value)
                        start_frame, end_frame = parse_framespan(framespan)
                        gt['Person'].append((start_frame, end_frame, height, width, x, y))
    return gt


def create_annotations(gt, camera_id, frame_shape):
    '''
    Convert ground truth from ISSIA dataset to SequenceAnnotations object
    Camera id and frame shape is needed to rectify the ball position (as for some strange reason actual ball position
    in ISSIA dataset is shifted/reversed for some sequences)
    For each frame we have:
    - list of ball positions (in ISSIA dataset there's only one ball, but in other datasets we can have more)
    - list of bounding boxes of players visible in this frame

    :param gt: dictionary with ISSIA ground truth data returned by load_groundtruth function
    :return: SequenceAnnotations object with ground truth data
    '''

    annotations = SequenceAnnotations()

    # In ISSIA dataset there's a discrepancy between frame number in the sequence and ground truth data
    # We need to add delta = 8 to the ground truth data, to get the real frame number (frame numbers start from 1)
    # delta = -8 works well for all sequences
    delta = -8
    for (start_frame, end_frame, x, y) in gt['BallPos']:
        for i in range(start_frame, end_frame+1):
            if camera_id == 2 or camera_id == 6:
                # For some reason ball coordinates for camera 2 and 6 have reversed in x-coordinate
                x = frame_shape[1] - x
            annotations.ball_pos[i+delta].append((x, y))

    for (start_frame, end_frame, height, width, x, y) in gt['Person']:
        assert start_frame <= end_frame
        for i in range(start_frame, end_frame+1):
            annotations.persons[i+delta].append((height, width, x, y))

    return annotations


def read_issia_ground_truth(camera_id, dataset_path):
    '''
    Read ground truth from the ISSIA dataset correpsoinding to camera: camera_id
    :param camera_id: number of teh sequence (between 1 and 6)
    :return: SequenceAnnotations object
    '''
    assert (camera_id >= 1) and (camera_id <= 6)

    dataset_path = os.path.expanduser(dataset_path)
    annotation_file = 'Film Role-0 ID-' + str(camera_id) + ' T-0 m00s00-026-m00s01-020.xgtf'
    annotation_filepath = os.path.join(dataset_path, annotation_file)

    gt = load_groundtruth(annotation_filepath)

    # This is needed to get frame size, so ISSIA ground truth can be rectified
    one_img_path = dataset_path+'/images/'+str(camera_id)+'0001.jpg'
    one_img = cv2.imread(one_img_path)

    annotations = create_annotations(gt, camera_id, one_img.shape)

    return annotations


def get_annotations(camera_id, idx):
    # Prepare annotations as list of boxes (label, x_top_left, y_top_left, width, height) in pixel coordinates
    boxes = []
    labels = []

    # Add annotations for the ball position: positions of the ball centre
    ball_pos = gt_annotations[camera_id].ball_pos[idx]

    for (x, y) in ball_pos:
        x1 = x - BALL_BBOX_SIZE // 2
        y1 = y - BALL_BBOX_SIZE // 2
        width = BALL_BBOX_SIZE
        height = BALL_BBOX_SIZE
        boxes.append((x1, y1, width, height))
        labels.append(BALL_LABEL)

    # Add annotations for the player position
    for (player_height, player_width, player_x, player_y) in gt_annotations[camera_id].persons[idx]:
        boxes.append((player_x, player_y, player_width, player_height))
        labels.append(PLAYER_LABEL)

    labels = np.column_stack((labels, boxes))
    return labels


def issis_format_to_yolo_format(xtl, ytl, w, h, frame):
    # return bounding box co_ordinates in (norm_x_center, norm_y_center, norm_width, norm_height) format
    img_h, img_w = frame[0], frame[1]
    xc = xtl + w/2
    yc = ytl + h/2

    norm_xc, norm_yc, norm_w, norm_h = xc/img_w, yc/img_h, w/img_w, h/img_h
    return norm_xc, norm_yc, norm_w, norm_h


def create_issia_dataset(dataset_path, cameras):
    # Get ISSIA datasets for multiple cameras
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    image_path, label_dir = os.path.join(dataset_path, 'images'), os.path.join(dataset_path, 'labels')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    assert os.path.exists(image_path), 'No such directory: ' + str(image_path)
    assert os.path.exists(label_dir), 'No such directory: ' + str(label_dir)

    for camera_id in cameras:
        # Extract frames from the sequence
        extract_frames(dataset_path, camera_id, image_path)

        # Read ground truth data for the sequence
        gt_annotations[camera_id] = read_issia_ground_truth(camera_id, dataset_path)

        # Create a list with ids of all images with any annotation
        annotated_frames = set(gt_annotations[camera_id].ball_pos) and set(gt_annotations[camera_id].persons)
        min_annotated_frame = min(annotated_frames)

        # Skip the first 50 annotated frames - as they may contain wrong annotations
        annotated_frames = [e for e in list(annotated_frames) if e > min_annotated_frame+50]

        image_list = []
        for e in annotated_frames:
            # Verify if the image file exists
            str_ = "{}{:04d}.jpg".format(camera_id,e)
            file_path = os.path.join(image_path, str_)
            if os.path.exists(file_path):
                image_list.append((file_path, camera_id, e))
            else:
                print(file_path, 'doesn\'t exist')

        frame = cv2.imread(image_list[0][0])

        total_count = 0
        frame_count = 0
        player_count = 0
        ball_count = 0

        for i_path, camera_id, e in image_list:
            if total_count%5==0:
                label_path = label_dir+'/'+i_path.split('/')[-1].replace(".jpg", ".txt")

                with open(label_path, "w") as out_file:
                    labels = get_annotations(camera_id, e)
                    for row in labels:
                        label, xtl, ytl, w, h = row[0], row[1], row[2], row[3], row[4]
                        if label==BALL_LABEL:
                            ball_count += 1
                        elif label==PLAYER_LABEL:
                            player_count += 1

                        norm_xc, norm_yc, norm_w, norm_h = issis_format_to_yolo_format(xtl, ytl, w, h, frame.shape)
                        line = str(label)+' '+str(norm_xc)+' '+str(norm_yc)+' '+str(norm_w)+' '+str(norm_h)

                        out_file.write(line)
                        out_file.write('\n')

                frame_count += 1
            total_count += 1

        print("{} frames, {} players, {} balls in camera {}".format(frame_count, player_count, ball_count, camera_id))

def remove_redundant_frames(images_dir, labels_dir):
    image_list = []
    for label in glob.glob(labels_dir+'/*.txt*'):
        image_list.append(label.split('/')[-1].replace('.txt', '.jpg'))
    count = 0
    for image in glob.glob(images_dir+'/*.jpg*'):
        img_name = image.split('/')[-1]
        if img_name in image_list:
            pass
        else:
            os.remove(image)

if __name__ == '__main__':
    dataset_path = '/home/sysadm/Desktop/soccer_analytics/dataset/ISSIA'

    cameras = [1, 2, 3, 4, 5, 6]
    create_issia_dataset(dataset_path, cameras)

    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    remove_redundant_frames(images_dir, labels_dir)
