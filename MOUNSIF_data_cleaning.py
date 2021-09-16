import os
import glob

BALL_LABEL = 0
PLAYER_LABEL = 1

def change_class_id(labels_path):
    assert os.path.exists(labels_path)

    for file in glob.glob(labels_path+'/*.txt'):
        file_ = file.replace('.txt', '_.txt')
        with open(file, 'r') as read_file, open(file_, 'w') as append_file:
            for line in read_file.readlines():
                if line != '\n':
                    line = line.split()
                    if line[0]=='1':
                        newline = str(BALL_LABEL)+' '+line[1]+' '+line[2]+' '+line[3]+' '+line[4]
                    elif line[0]=='2':
                        newline = str(PLAYER_LABEL)+' '+line[1]+' '+line[2]+' '+line[3]+' '+line[4]
                    append_file.write(newline)
                    append_file.write('\n')
            os.remove(file)

if __name__ == '__main__':
    dataset_path = '/home/sysadm/Desktop/soccer_analytics/dataset/MOUNSIF'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    labels_path = os.path.join(dataset_path, 'labels')
    change_class_id(labels_path)
