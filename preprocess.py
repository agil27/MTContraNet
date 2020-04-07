import os
import cv2
import csv
import h5py
import math
from tqdm import tqdm


def landmark_transition(mark, x, y, w, h):
    return [(m - x) * 96 / w if i % 2 == 0 else (m - y) * 96 / h for i, m in enumerate(mark)]


def main():
    affectnet_path = '/repository/lbj/affectnet'
    raw_path = os.path.join(affectnet_path, 'Manually_Annotated_Images')
    output_path = '/home/wangyuanbiao/affectnet'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def process_img(split):
        marking_path = os.path.join(affectnet_path, split + '.csv')
        pixels, labels, landmarks = [], [], []
        with open(marking_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader):
                label = int(row['expression'])
                if label >= 1 and label <= 6:
                    img_path = os.path.join(raw_path, row['subDirectory_filePath'])
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    Lx, Ly = int(row['face_x']), int(row['face_y'])
                    Rx, Ry = Lx + int(row['face_width']), Ly + int(row['face_height'])
                    landmark = row['facial_landmarks'].split(';')
                    landmark = [math.floor(float(l)) for l in landmark]
                    img = img[Ly:Ry, Lx:Rx]
                    img = cv2.resize(img, (96, 96))
                    pixels.append(img.tolist())
                    labels.append(label - 1)
                    landmark = landmark_transition(landmark, Lx, Ly, Rx - Lx, Ry - Ly)
                    landmarks.append(landmark)
                    # for i in range(len(landmark) // 2):
                    #     cv2.circle(img, (math.floor(landmark[2 * i]), math.floor(landmark[2 * i + 1])), 4, (0, 255, 0), -1)

            if len(labels) > 0:
                data_path = os.path.join(output_path, split + '.h5')
                with h5py.File(data_path, 'w') as datafile:
                    datafile.create_dataset('pixels', dtype='uint8', data=pixels)
                    datafile.create_dataset('labels', dtype='int32', data=labels)
                    datafile.create_dataset('landmarks', dtype='float32', data=landmarks)

    process_img('validation')
    process_img('training')


if __name__ == '__main__':
    main()
