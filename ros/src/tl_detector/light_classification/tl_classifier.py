import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self, is_site):
        # :: load classifier
        if is_site:
            GRAPH_FILE = r'light_classification/models/real_model/frozen_inference_graph.pb'
        else:
            GRAPH_FILE = r'light_classification/models/sim_model/frozen_inference_graph.pb'

        self.graph = self.create_graph(GRAPH_FILE)

        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.threshold = .5
        self.sess = tf.Session(graph=self.graph)
    
    def create_graph(self, graph_file):
        graph = tf.Graph()

        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')
        # with tf.Session(graph=tf.Graph()) as sess:
        #     tf.saved_model.loader.load(sess, ['serve'], graph_file)
        #     graph = tf.get_default_graph()
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # :: implement light color prediction
        img_expand = np.expand_dims(image, axis=0)
        with self.graph.as_default():
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if scores[0] > self.threshold:
            if classes[0] == 1:
                print("GREEN")
                return TrafficLight.GREEN
            elif classes[0] == 2:
                print("RED")
                return TrafficLight.RED
            elif classes[0] == 3:
                print("YELLOW")
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
