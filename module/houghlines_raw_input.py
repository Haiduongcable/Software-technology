import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
def detect_circle(img):
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #_, threshshold = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        x_list = []
        y_list = []
        r_list = []
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            x_list.append(x)
            y_list.append(y)
            r_list.append(r)
        rr = max(r_list)
        x_rr = x_list[r_list.index(rr)]
        y_rr = y_list[r_list.index(rr)]
        #cv2.circle(output, (x_rr, y_rr), rr, (0, 255, 0), 4)
        #cv2.rectangle(output, (x_rr - 5, y_rr - 5), (x_rr + 5, y_rr + 5), (0, 128, 255), -1)
    return x_rr, y_rr, rr

def threshold_rice(img):
    img_b = img[:,:,0]
    img_g = img[:,:,1]
    img_r = img[:,:,2]
    img_draw = img.copy()
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img_b[i,j] + img_g[i,j] + img_r[i,j] >= 100:
                img_draw[i,j] = [255,255,255]
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2GRAY)
    return img_draw
img_in = cv2.imread('/home/duongnh/Documents/Clock_time_project/img_raw_input_607.jpg')
img_draw = img_in.copy()
img_dr_1 = img_in.copy()
img_dr_2 = img_in.copy()
#minLineLength = 10
#maxLineGap = 10
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#lines = cv2.HoughLines(edges,1,np.pi/180, 200)
#lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
x_center, y_center, radius = detect_circle(img_in)

for i in range(np.shape(img_in)[0]):
    for j in range(np.shape(img_in)[1]):
        if math.sqrt(pow(j-x_center,2) + pow(i-y_center,2)) >= radius:
            img_draw[i,j] = [255,255,255]
img_test_output = img_draw.copy()
def calculate_angle(u):
    v = np.array([0,10])
    c = dot(u, v) / norm(u) / norm(v)  # -> cosine of the angle
    angle = arccos(clip(c, -1, 1)) / np.pi * 180
    return angle
'''
def get_only_2_vecto_line(img_test_output_input_def,x_center):
    img_out = img_test_output_input_def.copy()
    gray = cv2.cvtColor(img_test_output_input_def, cv2.COLOR_BGR2GRAY)
    # thresh = threshold_rice(img_in)
    _, thresh_shold_1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh_shold_1, 20, 155, apertureSize=3)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    list_s_line = []
    list_e_line = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            list_s_line.append((l[0],l[1]))
            list_e_line.append((l[2],l[3]))
    list_s_out = list_s_line.copy()
    list_e_out = list_e_line.copy()
    for i_s in range(len(list_s_line) - 1):
        for i_e in range(i_s + 1,len(list_s_line)):
            if math.sqrt(pow(list_s_line[i_s][0]-list_s_line[i_e][0],2) +
                         pow(list_s_line[i_s][1] - list_s_line[i_e][1],2)) <= 10 and\
                    math.sqrt(pow(list_e_line[i_s][0]-list_e_line[i_e][0],2) +
                              pow(list_e_line[i_s][1] - list_e_line[i_e][1],2)) <= 10:
                list_s_out.pop(i_s)
                list_e_out.pop(i_s)

    if len(list_s_out) == 1:
        u = np.array([list_e_out[0][0] - list_s_out[0][0],
                      list_e_out[0][1] - list_s_out[0][1]])
        angle = calculate_angle(u)
        if list_s_out[0][0] >= x_center:
            hour = (angle/180)*6
            minute = (angle/180)*30
        else:
            hour = 12 - (angle/180)*6
            minute = 60 - (angle/180)*30
        return hour, minute
    if len(list_s_out) == 2:
        len_list = []
        for i in range(len(list_s_out)):
            len_list.append(math.sqrt(pow(list_e_out[i][0] - list_s_out[i][0],2) +
                                      pow(list_e_out[i][1] - list_s_out[i][1],2) ))
        kim_phut = max(len_list)
        u_minute = np.array([list_e_out[len_list.index(kim_phut)][0] -
                             list_s_out[len_list.index(kim_phut)][0]
                                , list_e_out[len_list.index(kim_phut)][1] -
                             list_s_out[len_list.index(kim_phut)][1]])

        u_hour = np.array([list_e_out[1 - len_list.index(kim_phut)][0] -
                           list_s_out[1 - len_list.index(kim_phut)][0]
                                , list_e_out[1 - len_list.index(kim_phut)][1] -
                           list_s_out[1 - len_list.index(kim_phut)][1]])
        angle_minute = calculate_angle(u_minute)
        angle_hour = calculate_angle(u_hour)
        if list_s_out[len_list.index(kim_phut)][0] >= x_center:
            minute = minute = (angle_minute/180)*30
        else:
            minute = 60 - (angle_minute/180)*30
        if list_s_out[1 - len_list.index(kim_phut)][0] >= x_center:
            hour = hour = (angle_hour/180)*6
        else:
            hour = 12 - (angle_hour/180)*6
        return hour, minute
hour_raw,minute_raw = get_only_2_vecto_line(img_test_output,x_center)
if hour_raw >= int(hour_raw) + 1 \
        and hour_raw <= int(hour_raw) + 1.2 \
        and minute_raw >= 45:
    hour = int(hour_raw) - 1
    minute = int(minute_raw)
else:
    hour = int(hour_raw)
    minute = int(minute_raw)
'''








