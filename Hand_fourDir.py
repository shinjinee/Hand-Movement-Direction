import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.2, help='Score threshold for displaying bounding boxes')
parser.add_argument('-fps', '--fps', dest='fps', type=int, default=1, help='Show FPS on detection/display visualization')
parser.add_argument('-src', '--source', dest='video_source', default=0, help='Device index of the camera.')
parser.add_argument('-wd', '--width', dest='width', type=int, default=320, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int, default=180, help='Height of the frames in the video stream.')
parser.add_argument('-ds', '--display', dest='display', type=int, default=1, help='Display the detected images using OpenCV. This reduces FPS')
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=4, help='Number of workers.')
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=5, help='Size of the queue.')
args = parser.parse_args()


# load frozen tensorflow model into memory
def load_inference_graph():
	print("> ====== loading HAND frozen graph into memory")
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile('hand_inference_graph/frozen_inference_graph.pb', 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
		sess = tf.compat.v1.Session(graph=detection_graph)
	print(">  ====== Hand Inference graph loaded.")
	return detection_graph, sess
	
	
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
	# Each box represents a part of the image where a particular object was detected.
	# Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name( 'detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)

def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width, boxes[i][0] * im_height, boxes[i][2] * im_height)
            
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
def draw_left(image_np):
	cv2.putText(image_np, "Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	
def draw_right(image_np):
	cv2.putText(image_np, "Right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	
def draw_up(image_np):
	cv2.putText(image_np, "Up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

def draw_down(image_np):
	cv2.putText(image_np, "Down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


detection_graph, sess=load_inference_graph()

cap = cv2.VideoCapture(args.video_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

start_time = datetime.datetime.now()
num_frames = 0
im_width, im_height = (cap.get(3), cap.get(4))
num_hands_detect = 1
coun=0
(left1, right1, top1, bottom1) = (0, 0, 0, 0)
cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

while True:

	if (coun==5):
		coun=0
		
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	ret, image_np = cap.read()
	image_np=cv2.flip(image_np, 1)
	try:
		image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
	except:
		print("Error converting to RGB")

	num_frames += 1
	elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
	fps = num_frames / elapsed_time
	
	boxes, scores = detect_objects(image_np, detection_graph, sess)
	draw_box_on_image(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)
	
	if coun==0:
		(left1, right1, top1, bottom1) = (boxes[0][1] * im_width, boxes[0][3] * im_width, boxes[0][0] * im_height, boxes[0][2] * im_height)
		
	if coun==4:
		(left2, right2, top2, bottom2) = (boxes[0][1] * im_width, boxes[0][3] * im_width, boxes[0][0] * im_height, boxes[0][2] * im_height)
		print (left2, bottom2, right2, top2)
		
		if (abs(bottom2-bottom1)<(0.1*bottom1) and abs(top2-top1)<(0.1*top1) and (left1-left2)>(0.2*left1) and (right1-right2)>(0.2*right1)):
			draw_left(image_np)
			
		if (abs(bottom2-bottom1)<(0.1*bottom1) and abs(top2-top1)<(0.1*top1) and (right2-right1)>(0.2*right1) and (left2-left1)>(0.2*left1)):
			draw_right(image_np)
			
		if (abs(right2-right1)<(0.1*right1) and abs(left2-left1)<(0.1*left1) and (top1-top2)>(0.2*top1) and (bottom1-bottom2)>(0.2*bottom1)):
			draw_up(image_np)
			
		if (abs(right2-right1)<(0.1*right1) and abs(left2-left1)<(0.1*left1) and (bottom2-bottom1)>(0.2*bottom1) and (top2-top1)>(0.2*top1)):
			draw_down(image_np)
			
	coun=coun+1
	#draw_fps_on_image("FPS : " + str(int(fps)), image_np)
	cv2.imshow('Single-Threaded Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
			
	if cv2.waitKey(25) & 0xFF == ord('q'):
				break
            	
cv2.destroyAllWindows()
