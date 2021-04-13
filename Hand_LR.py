import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.5, help='Score threshold for displaying bounding boxes')
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

	#for op in detection_graph.get_operations():
	#	print(op.values())

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

def draw_box_on_image(score_thresh, scores, boxes, im_width, im_height, image_np):
	if (scores[0] > score_thresh):
		(left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width, boxes[0][0] * im_height, boxes[0][2] * im_height)
		p1 = (int(left), int(top))
		p2 = (int(right), int(bottom))
		cv2.rectangle(image_np, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)
    
def draw_limits(image_np):
	cv2.line(image_np, (int(round(0.375*image_np.shape[1])),0), (int(round(0.375*image_np.shape[1])),image_np.shape[0]), (0,0,0), 1)
	cv2.line(image_np, (int(round(0.625*image_np.shape[1])),0), (int(round(0.625*image_np.shape[1])),image_np.shape[0]), (0,0,0), 1)
                
def draw_left(image_np, limL, left, top):
	per = ((limL-left) / limL) * 100
	cv2.putText(image_np, "Left: "+str(round(per, 2))+"%", (int(round(left-5)), int(round(top-3))), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
	
def draw_right(image_np, limR, w, right, top, left):
	per = ((right-limR) / (w-limR)) * 100
	cv2.putText(image_np, "Right: "+str(round(per, 2))+"%", (int(round(left-5)), int(round(top-3))), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)


detection_graph, sess=load_inference_graph()

cap = cv2.VideoCapture(args.video_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

im_width, im_height = (cap.get(3), cap.get(4))

cv2.namedWindow('Left-Right_percent', cv2.WINDOW_NORMAL)

while True:
		
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	ret, image_np = cap.read()
	image_np=cv2.flip(image_np, 1)
	h,w,c = image_np.shape
	try:
		image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
	except:
		print("Error converting to RGB")
	
	boxes, scores = detect_objects(image_np, detection_graph, sess)
	
	draw_box_on_image(args.score_thresh, scores, boxes, im_width, im_height, image_np)
	draw_limits(image_np)
	
	(left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width, boxes[0][0] * im_height, boxes[0][2] * im_height)
	
	limL, limR = (w/2)-(0.125*w), (w/2)+(0.125*w)
	if (scores[0] > args.score_thresh):
	
		if (left<limL and right>limR):
			print ("Error")
		
		elif (right<limR and left<limL):
			draw_left(image_np, limL, left, top)
		
		elif (right>limR and left>limL):
			draw_right(image_np, limR, w, right, top, left)
	
	cv2.imshow('Left-Right_percent', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
			
	if cv2.waitKey(25) & 0xFF == ord('q'):
				break
            	
cv2.destroyAllWindows()
