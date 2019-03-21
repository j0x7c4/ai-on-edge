import cv2 as cv
import sys
import logging
import time

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)

# Load the model
t = time.time()
net = cv.dnn.readNet('models/hello/face-detection-adas-0001.xml', 'models/hello/face-detection-adas-0001.bin')
logging.info("load model cost %f" % (time.time() - t))
# Specify target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# Read an image
frame = cv.imread(sys.argv[1])

# Prepare input blob and perform an inference
t = time.time()
blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)
out = net.forward()
logging.info("inference cost %f" % (time.time() - t))

# Draw detected faces on the frame
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])

    if confidence > 0.5:
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

# Save the frame to an image file
cv.imwrite('out.png', frame)
