import cv2
import numpy as np
import math
from numpy.matlib import repmat

pi = math.pi

# Flatten image from Polar to decartes


def mapping_coordinates(x_cen=1238, y_cen=1040, R=710, r=470):
    # Kich thuoc anh sau khi flat co do dai bang chu vi duong tron
    width = int(2 * pi * R)

    # Do rong cua duong vanh khan chua san pham (chieu cao anh flatten)
    height = R - r

    # cong them 400 de template matching
    width_padding = width

    # doi chieu tu he toa do Decartes sang toa do cuc
    # Lay vi cua cac pixels can cat trong toa do cuc tao ma tran temp de flat
    temp = R - np.array(range(height))
    temp = temp.reshape(height, 1)
    temp = repmat(temp, 1, width_padding)

    # Tinh toan goc alpha
    alpha = pi / 2 - np.array(range(width_padding)) / width * 2 * pi
    alpha = alpha.reshape(1, width_padding)
    alpha = repmat(alpha, height, 1)

    # Chuyen doi tu toa do cuc sang toa do Decarc
    x_mapping = np.round(x_cen + temp * np.cos(alpha)).astype(int)
    y_mapping = np.round(y_cen - temp * np.sin(alpha)).astype(int)
    print(x_mapping, y_mapping)
    # return mapping between polar and decartes, height, width, padding width of image
    return x_mapping, y_mapping, height, width, width_padding

if __name__ == "__main__":
    
    x_mapping, y_mapping, h, w, p = mapping_coordinates(442, 236, 200, 0)

    # Read until video is completed
    index = 0
    size = (p, h)
    out = cv2.VideoWriter("result.avi",cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
    cap = cv2.VideoCapture('Clock.mp4')
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            flatten = frame[y_mapping[:, :], x_mapping[:, :]]
            cv2.imwrite("Video Frame/{}.jpg".format(index), flatten)
            out.write(flatten)
            index+=1
        else:
            break
    cap.release()
    out.release()
