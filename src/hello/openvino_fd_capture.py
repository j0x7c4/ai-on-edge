import cv2 as cv
import sys
import logging
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
from utils.PCA9685 import PCA9685
from queue import Queue
from threading import Thread

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)

# Load the model
t = time.time()
net = cv.dnn.readNet('models/hello/face-detection-adas-0001.xml', 'models/hello/face-detection-adas-0001.bin')
logging.info("load model cost %f" % (time.time() - t))
# Specify target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

screen_size = (640, 480)

def run_camera(q):
    # Load Camera
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
     
    # allow the camera to warmup
    time.sleep(0.1)
     
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        
        # Prepare input blob and perform an inference
        t = time.time()
        blob = cv.dnn.blobFromImage(image, size=(672, 384), ddepth=cv.CV_8U)
        net.setInput(blob)
        out = net.forward()
        logging.debug("inference cost %f" % (time.time() - t))

        # Draw detected faces on the image
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * image.shape[1])
            ymin = int(detection[4] * image.shape[0])
            xmax = int(detection[5] * image.shape[1])
            ymax = int(detection[6] * image.shape[0])

            x_mid = (xmin+xmax)/2
            y_mid = (ymin+ymax)/2
            if confidence > 0.5:
                if ymin > image.shape[1]/2:
                    q.put("down")
                if ymax < image.shape[1]/2:
                    q.put("up")
                if xmin > image.shape[0]/2:
                    q.put("right")
                if xmax < image.shape[0]/2:
                    q.put("left")
                logging.debug("xmin=%s, ymin=%s, xmax=%s, ymax=%s"%(xmin, ymin, xmax, ymax))
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

        # Save the frame to an image file
        # cv.imwrite('out.png', image)

        # show the frame
        cv.imshow("Frame", image)
        key = cv.waitKey(1) & 0xFF
        logging.debug("key=%s" % key)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == 81:
            q.put("left")
        if key == 82:
            q.put("up")
        if key == 83:
            q.put("right")
        if key == 84:
            q.put("down")
        if key == ord("q"):
            q.put("exit")
            break

def run_move(q):
    channel_yaw = 7
    channel_pitch = 15
    pos_yaw = 1400
    pos_pitch = 1400
    default_step = 100
    max_pos = 2400
    min_pos = 600
    # init pwm
    pwm = PCA9685(0x40)
    pwm.setPWMFreq(50)
    pwm.setServoPulse(channel_yaw, pos_yaw)
    pwm.setServoPulse(channel_pitch, pos_pitch)

    while True:
        data = q.get()
        logging.info("move=%s" % str(data))
        if data == 'exit':
            break
        elif data == "left":
            pos_yaw += default_step
            pos_yaw = min(max(min_pos, pos_yaw), max_pos)
            pwm.setServoPulse(channel_yaw, pos_yaw)
        elif data == "right":
            pos_yaw -= default_step
            pos_yaw = min(max(min_pos, pos_yaw), max_pos)
            pwm.setServoPulse(channel_yaw, pos_yaw)
        elif data == "up":
            pos_pitch += default_step
            pos_pitch = min(max(min_pos, pos_pitch), max_pos)
            pwm.setServoPulse(channel_pitch, pos_pitch)
        elif data == "down":
            pos_pitch -= default_step
            pos_pitch = min(max(min_pos, pos_pitch), max_pos)
            pwm.setServoPulse(channel_pitch, pos_pitch)
        logging.debug("pos_yaw=%s,pos_pitch=%s" % (pos_yaw, pos_pitch))

q = Queue()
t1 = Thread(target=run_move, args=(q,))
t2 = Thread(target=run_camera, args=(q,))
t1.start()
t2.start()
