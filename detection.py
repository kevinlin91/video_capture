# coding:utf8
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import urllib
import tarfile
from utils import label_map_util
from utils import visualization_utils as vis_util



class object_detection(object):


    def __init__(self):
        
        if not os.path.isdir('./ssd_mobilenet_v1_coco_2017_11_17'):
            self.download()

        
        self.NUM_CLASSES = 90
        self.PATH_TO_CKPT = './ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
        self.PATH_TO_TEST_IMAGES_DIR = './mushroom.jpg'
        self.PATH_TO_LABELS = os.path.join('models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
        self.detection_graph = self.load_model()
        self.category_index = self.load_label_map()


    def download(self):
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'                
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    def load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return detection_graph
        
    def load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image, with_video=0):
        with self.detection_graph.as_default():
          with tf.Session(graph=self.detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_np_expanded = np.expand_dims(image, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
              # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                  image,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  self.category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
            if with_video==0:
                while True:
                    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
                    cv2.imshow("detection", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        return image                    
        
    



if __name__=='__main__':
    image = cv2.imread('./mushroom.jpg')
    obj_detect = object_detection()
    image = obj_detect.detect(image,with_video=1)
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    cv2.imshow('detection',image)
    cv2.imwrite('output.png',image)
