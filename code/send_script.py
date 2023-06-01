#!/usr/bin/env python

import rclpy
import os
import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *

import cv2
import numpy as np
from math import pi,tan
import math
import matplotlib.pyplot as plt
from copy import deepcopy


def binarize(image , threshold):
    bin_image=cv2.threshold(image, threshold, 255,cv2.THRESH_BINARY)[1]
    return bin_image

# arm client
def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

# gripper client
def set_io(state):
    gripper_node = rclpy.create_node('gripper')
    gripper_cli = gripper_node.create_client(SetIO, 'set_io')

    while not gripper_cli.wait_for_service(timeout_sec=1.0):
        gripper_node.get_logger().info('service not availabe, waiting again...')

    io_cmd = SetIO.Request()
    io_cmd.module = 1
    io_cmd.type = 1
    io_cmd.pin = 0
    io_cmd.state = state
    gripper_cli.call_async(io_cmd)
    gripper_node.destroy_node()
    # print("io in")

def getSingleChessCenter(image, rough_cx: int, rough_cy: int) -> list:
	temp_image = image.copy()

	row, col = image.shape
	# cv2.imshow("roi", temp_image)
	# cv2.waitKey(0)

	length = 100
	width = 100

	new_width =  min(rough_cy*2, width)
	new_length = min(rough_cx*2, length)


	crop_temp_image = temp_image[rough_cy-new_width//2:rough_cy+new_width//2,
                                 rough_cx-new_length//2:rough_cx+new_length//2]

	rows = crop_temp_image.shape[0]
	# detect circles in the image
	circles = cv2.HoughCircles(crop_temp_image, cv2.HOUGH_GRADIENT, 1, rows / 8,
								param1=100, param2=30,
								minRadius=10, maxRadius=20)

	result = []

	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			result.append((x+(rough_cx-new_length//2), y+(rough_cy-new_width//2)))

			cv2.circle(temp_image, (x+(rough_cx-new_length//2), y+(rough_cy-new_width//2)), r, (0, 255, 0), 4)
			cv2.circle(temp_image, (x+(rough_cx-new_length//2), y+(rough_cy-new_width//2)), 1, (0, 0, 255), 4)

	# cv2.imshow("temp_image", temp_image)
	# cv2.waitKey(0)

	return result

def rotation(x,y,angle):
    angle = angle * (pi / 180)
    # print((x*cos(angle)+y*sin(angle), x*(-1)*sin(angle)+y*cos(angle)))
    return np.array((x*math.cos(angle)+y*math.sin(angle), x*(-1)*math.sin(angle)+y*math.cos(angle)))

def division(start,end,number):
    dx = (end[0] - start[0] )/number
    dy = (end[1] - start[1] )/number
    return [(start[0]+i*dx,start[1]+i*dy) for i in range(number)] + [end]

def read_text_file(textFileName):
    try:
        f = open(textFileName, "r")
        latest_f_read = f.read()
        latest_command = latest_f_read.split("\n")[-2]
        commands = latest_command.split(":")[1].split("->")

        from_position = int(commands[0])
        to_position = int(commands[1])
        action = commands[2]
    except:
        return None

    return from_position, to_position, action

def main(args=None):

    # XY = 230, 230 BLUE
    # XY = 400, 153 ORANGE
    # XY = 235, 400 GREEN
    # XY = 470, 360 YELLOW
    # real_x_c = [230, 400, 235, 470]
    # real_y_c = [230, 153, 400, 360]

    rclpy.init(args=args)
    # targetP1 = "100.00 , 150.00 , 400.00 , 180.00 , 0.00 , 0.00"
    # targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    set_io(0.0)
    targetP1 = "230.00, 230, 800, -180.00, 0.0, 135.00"
    script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
    send_script(script)
    # set_io(0.0)

    send_script("Vision_DoJob(job1)")

    imagePathEmpty = "/home/robotics/workspace2/team6_ws/result_chess_empty.png"

    imagePath = "/home/robotics/workspace2/team6_ws/result_chess.png"
    if os.path.isfile(imagePath):
        os.remove(imagePath)

    # image that contain chess piece
    imgWithChess = None
    while imgWithChess is None:
        imgWithChess = cv2.imread(imagePath)

    imgWithChess = cv2.cvtColor(imgWithChess, cv2.COLOR_BGR2GRAY)

    '''
        Below code: Get every single coordinate of chess board
    '''
    img = cv2.imread(imagePathEmpty)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)

    ret, binarized = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    binarized = cv2.bitwise_not(binarized)

    contours, _ = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boardContours = np.zeros(binarized.shape[:2], dtype='uint8')
    cv2.drawContours(boardContours, contours, -1, (255, 0, 0), 1)
    # cv2.imshow("boardContours", boardContours)
    # cv2.waitKey(0)

    corners = []
    pose = []
    board = np.zeros(binarized.shape[:2], dtype='uint8')

    # find minimum rectangle
    for i in contours:
        (cx, cy), (width, height), angle = rect = cv2.minAreaRect(i)

        if width*height > 500000:
            box =  cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(board, [box], -1, (255, 0, 0), 1)
            cv2.circle(board, (int(cx), int(cy)), 7, (255), 1)

            pose.append(((cx, cy), (width, height), angle))

    (cx, cy), (width, height), angle = pose[-1]

    if width>height:
        height,width= max(height,width),min(width,height)
        angle = 90 - angle
    k = width/82

    #棋盤上的座標順時針旋轉90-angle 即為 image座標
    center = np.array([cx,cy])
    corners.append(center+ rotation((height/2)-3*k, (width/2)-k, 90-angle)) # 右下
    corners.append(center+ rotation((height/2)-3*k, -(width/2)+k, 90-angle)) # 右上
    corners.append(center+ rotation(-(height/2)+3*k, -(width/2)+k, 90-angle)) # 左上
    corners.append(center+ rotation(-(height/2)+3*k, (width/2)-k, 90-angle)) #左下

    corners = [(int(array[0]),int(array[1])) for array in corners]
    corner_img = deepcopy(board)
    for corner in corners:
        cv2.circle(corner_img, corner, 7, (255), 1)
    cv2.imwrite("./corner.png", corner_img)

    # chessboard coordinate
    coordinates = []

    # find every single point of the chessboard
    edges = [division(corners[0],corners[1],8),division(corners[1],corners[2],9),division(corners[2],corners[3],8),division(corners[3],corners[0],9)]
    for idx, start in enumerate(edges[0]):
        coordinates.extend(division(start,edges[2][-1*idx-1],9))
    coordinate_img = deepcopy(board)
    for coordinate in coordinates:
        cv2.circle(coordinate_img, (int(coordinate[0]), int(coordinate[1])), 7, (255), 1)
    cv2.imwrite("/home/robotics/workspace2/team6_ws/coordinate.png", coordinate_img)

    '''
        Get position from text file
    '''
    from_pos, to_pos, action = read_text_file("/home/robotics/workspace2/team6_ws/src/send_script/send_script/command.txt")
    if from_pos == None:
        print("from_pos Missing")
        return

    print("Will go to grab piece {0} to piece {1}".format(from_pos, to_pos))

    '''
        Below code: Find single piece coordinate and grab it
    '''

    COORD_X = coordinates[from_pos][0]
    COORD_Y = coordinates[from_pos][1]

    to_COORD_X = coordinates[to_pos][0]
    to_COORD_Y = coordinates[to_pos][1]

    roughCx = int(COORD_X)
    roughCy = int(COORD_Y)

    chessPieces = getSingleChessCenter(imgWithChess, roughCx, roughCy)

    # take out the piece to be killed
    if action == "eat":
        chessPiecesToBeKilled = getSingleChessCenter(imgWithChess, int(to_COORD_X), int(to_COORD_Y))
        x =  - 0.395365224 * chessPiecesToBeKilled[0][1] + 0.403578791 * chessPiecesToBeKilled[0][0] + 237.607317
        y =  - 0.391017121 * chessPiecesToBeKilled[0][1] - 0.397007041 * chessPiecesToBeKilled[0][0] + 742.490986
        angle = 45
        print("X: {0}, Y: {1}, Angle: {2}".format(x, y, angle))

        print("to_pos", to_pos)

        if to_pos %10 == 0:
            x += 7
        elif to_pos % 10 == 1:
            x += 6
        elif to_pos % 10 == 2:
            x += 5
        elif to_pos % 10 == 9:
            x -= 2
            y += 2


        targetP1 = str(x) + "," + str(y) + ", 140.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        targetP1 = str(x) + "," + str(y) + ", 100.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        set_io(1.0)

        targetP1 = str(x) + "," + str(y) + ", 140.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        # TODO: GO TO DUMP POSITION
        targetP1 = str(800) + "," + str(300) + ", 140.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        set_io(0.0)


    for i in range(1):
        x =  - 0.395365224 * chessPieces[0][1] + 0.403578791 * chessPieces[0][0] + 237.607317
        y =  - 0.391017121 * chessPieces[0][1] - 0.397007041 * chessPieces[0][0] + 742.490986
        angle = 45
        print("X: {0}, Y: {1}, Angle: {2}".format(x, y, angle))

        if from_pos % 10 == 0:
            x += 7
        elif from_pos % 10 == 1:
            x += 6
        elif from_pos % 10 == 2:
            x += 5
        elif from_pos % 10 == 9:
            x -= 2
            y += 2


        targetP1 = str(x) + "," + str(y) + ", 140.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        targetP1 = str(x) + "," + str(y) + ", 100.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        set_io(1.0)

        # targetP1 = str(x) + "," + str(y) + ", 100.00 , 180.00 , 0.00 , 135.00"
        # script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        # send_script(script)

        targetP1 = str(x) + "," + str(y) + ", 140.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)


    for i in range(1):
        x =  - 0.395365224 * to_COORD_Y + 0.403578791 * to_COORD_X + 237.607317
        y =  - 0.391017121 * to_COORD_Y - 0.397007041 * to_COORD_X + 742.490986
        angle = 45
        print("xOr: {0}, yOr: {1}, Angle: {2}".format(x, y, angle))

        if to_pos %10 == 0:
            x += 7
        elif to_pos % 10 == 1:
            x += 6
        elif to_pos % 10 == 2:
            x += 5
        elif to_pos % 10 == 9:
            x -= 0
            y += 8


        targetP1 = str(x) + "," + str(y) + ", 140.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        targetP1 = str(x) + "," + str(y) + ", 100.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        set_io(0.0)

        targetP1 = str(x) + "," + str(y) + ", 400.00 , 180.00 , 0.00 , 135.00"
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)

        # targetP1 = str(x) + "," + str(y) + ", 105.00 , 180.00 , 0.00 , 135.00"
        # script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        # send_script(script)




    rclpy.shutdown()


if __name__ == '__main__':
    main()