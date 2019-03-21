import cv2 as cv
import sys
import logging
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)

# Load the model
t = time.time()
net = cv.dnn.readNet('models/hello/face-detection-adas-0001.xml', 'models/hello/face-detection-adas-0001.bin')
logging.info("load model cost %f" % (time.time() - t))
# Specify target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

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
    logging.info("inference cost %f" % (time.time() - t))

    # Draw detected faces on the image
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * image.shape[1])
        ymin = int(detection[4] * image.shape[0])
        xmax = int(detection[5] * image.shape[1])
        ymax = int(detection[6] * image.shape[0])

        if confidence > 0.5:
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

    # Save the frame to an image file
    # cv.imwrite('out.png', image)

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break