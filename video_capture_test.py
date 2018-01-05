from detection import object_detection
from utils import visualization_utils as vis_util
import tensorflow as tf
import numpy as np
import cv2



#webcam capture
#cap = cv2.VideoCapture(1)

#video from disk
cap = cv2.VideoCapture('./IpfMyc2OrSs.avi')
detection = object_detection()

def convert_gray_video():
  while(True):
    ret, frame = cap.read()
    #test color to gray
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
  cap.release()
  
def convert_object_detection():
  with detection.detection_graph.as_default():
    with tf.Session(graph=detection.detection_graph) as sess:
      while True:
        ret, frame = cap.read()
        image_np_expanded = np.expand_dims(frame, axis=0)
        image_tensor = detection.detection_graph.get_tensor_by_name('image_tensor:0')
        
        boxes = detection.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection.detection_graph.get_tensor_by_name('num_detections:0')
  
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
      
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            detection.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('frame',cv2.resize(frame,(800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
      
if __name__=='__main__':
  #convert_gray_video()
  convert_object_detection()
