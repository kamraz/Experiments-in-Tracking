import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops

from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2 as cv
import argparse

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

'''
This class is for tensorflow object detection API
See there for more information and documentation
'''
class Detection:
    def __init__(self):
        pass
    def download(self):
        # What model to download.
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
        return PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS

    def loadGraph(self, path, labels):
        PATH_TO_FROZEN_GRAPH = path
        PATH_TO_LABELS = labels
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        return detection_graph, category_index

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

#Below code opens video
parser = argparse.ArgumentParser(description='Detect and Optical Flow')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)
detect = Detection()

#First trial this line must be uncommented to download model
#sysGraph, sysLabels = detect.download()

#Loads model graph
sysGraph = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
sysLabels = 'data/mscoco_label_map.pbtxt'
print(sysGraph)
print(sysLabels)
graph, category = detect.loadGraph(sysGraph, sysLabels)

#Loads frame and runs object detection on that frame
ret, old_frame = cap.read()
size = old_frame.shape
image_np = np.asarray(old_frame)
image_np_expanded = np.expand_dims(image_np, axis=0)
output_dict = detect.run_inference_for_single_image(image_np_expanded, graph)

#Visualizes detections saves image, then prints them to terminal
vis_util.visualize_boxes_and_labels_on_image_array(
  image_np,
  output_dict['detection_boxes'],
  output_dict['detection_classes'],
  output_dict['detection_scores'],
  category,
  instance_masks=output_dict.get('detection_masks'),
  use_normalized_coordinates=True,
  line_thickness=2)
plt.figure(figsize=(12,8))
plt.imshow(image_np)
print(output_dict)

#Finds the car with the highest confidence
for index, value in enumerate(output_dict['detection_boxes']):
    if output_dict['detection_classes'][index] == 3 and output_dict['detection_scores'][index]>=.5:
        y1 = int(value[0]*size[0])
        x1 = int(value[1]*size[1])
        y2 = int(value[2]*size[0])
        x2 = int(value[3]*size[1])
        print(x1)
        print(y1)
        print(x2)
        print(y2)
        break
plt.savefig('myfilename.png', dpi=100)

#Shrink x and y to fit car better
x1 = int(x1 +(x2-x1)*.2)
x2 = int(x2 -(x2-x1)*.2)
y1 = int(y1 +(y2-y1)*.2)
y2 = int(y2 -(y2-y1)*.2)

#Now initializes trackers
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
mask = np.zeros(shape = old_gray.shape, dtype = "uint8")
cv.rectangle(mask,(x1,y1),(x2,y2),(255,255,255),-1)
cv.imshow('msk', mask)

#Finds good points to track
p0 = cv.goodFeaturesToTrack(old_gray,4,.01,5, mask=mask)

lk_params = dict( winSize  = (15,15),
                  maxLevel = 10,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

#run video with trackers
mask = np.zeros_like(old_frame)
color = np.random.randint(0,255,(100,3))
while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
