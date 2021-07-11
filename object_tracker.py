import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Import dependencies    
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

'''
absl.flags defines a distributed command line system. Rather than an application
having to define all flags in or near main(), each Python module defines flags
that are useful to it. When one Python module imports another, it gains access
to the other’s flags.

The Abseil flags library includes the ability to define flag types (boolean,
float, integer, list), autogeneration of help (in both human and machine
readable format) and reading arguments from a file. It also includes the
ability to automatically generate manual pages from the help flags.

Flags are defined through the use of DEFINE_* functions (where the flag’s type
is used to define the value).

Source : [https://abseil.io/docs/python/guides/flags]
'''
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('output_data', False, 'path to output data')

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def main(_argv):
    # Define parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    # initialize frame to be able to loop over video frames
    frame_num = 0

    # It's going to contain the thracking history
    memory = {}

   
    
   
    '''
    # video après midi
    line1 = [(190, 370),(460,370)]#IN
    line2 = [(155, 575),(890,575)]#OUT
    line3 = [(360, 450),(670,450)]#Jawaz
    line4 = [(730, 430), (990, 430)]#Manual  
    '''
    '''
    # video nuit
    line1 = [(0, 150), (170, 150)]#IN
    line2 = [(0, 285), (430, 285)]#OUT
    line3 = [(230, 160), (385, 160)]#Jawaz
    line4 = [(435, 135), (550, 135)]#Manual  
    '''
    #initialize line parameters
    line1 = [(190, 370),(460,370)]#IN
    line2 = [(155, 575),(890,575)]#OUT
    line3 = [(360, 450),(670,450)]#Jawaz
    line4 = [(610, 500),(915,500)]#Manual

    #initialize counters
    counter_IN = 0
    counter_OUT = 0
    counter_JAWAZ = 0
    counter_MANUEL = 0
    moyenne_fps = 0

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file (uncomment line below to track all default classes)
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker)
        allowed_classes = ['car','truck']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        # update tracks
        tracker.update(detections)

        # keep valid boxes parameters in boxes 
        boxes = []
        # keep objects' indexes in indexIDs
        indexIDs = []
        # previous serves to record previous memory of tracked objects 
        previous = memory.copy()
        memory = {}

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            indexIDs.append(int(track.track_id))
            memory[indexIDs[-1]] = boxes[-1]
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        
        # Counting objects
        if len(boxes) > 0:
            i = int(0)
            for box in boxes :
               # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                if indexIDs[i] in previous:
                    
                    previous_box = previous[indexIDs[i]]
                    # extract previous bounding box coordinates
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))

                    # The center of the current box
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2)) 

                    # The center of the previous box
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))

                    # draw the tracking line
                    cv2.line(frame, p0, p1, color, 3)

                    '''if there is ever an intersection between one of the lines 
                    and the vehicle tracking line, the counter is incremented'''

                    if intersect(p0, p1, line1[0], line1[1]):
                        counter_IN += 1
                    if intersect(p0, p1, line2[0], line2[1]):
                        counter_OUT += 1
                    if intersect(p0, p1, line3[0], line3[1]):
                        counter_JAWAZ += 1
                    if intersect(p0, p1, line4[0], line4[1]):
                        counter_MANUEL += 1     
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                i += 1

        # draw line
        cv2.line(frame, line1[0], line1[1], (0, 0, 0xFF), 2)
        cv2.line(frame, line2[0], line2[1], (0, 0, 0xFF), 2)
        cv2.line(frame, line3[0], line3[1], (0, 0, 0xFF), 2)
        cv2.line(frame, line4[0], line4[1], (0, 0, 0xFF), 2)
        
        # draw counter
        cv2.putText(frame,"IN: {}".format(counter_IN),(5, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "OUT:{}".format(counter_OUT), (5, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "JAWAZ:{}".format(counter_JAWAZ), (5, 175), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "MANUAL:{}".format(counter_MANUEL), (5, 245), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
   
        '''
        cv2.putText(frame, str(counter_IN), (90, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, str(counter_OUT), (130, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2) 
        cv2.putText(frame, str(counter_JAWAZ), (160, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, str(counter_MANUEL), (200, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        '''
        # counter += 1      
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        elap = (time.time() - start_time)
        moyenne_fps = moyenne_fps + fps
        print("FPS: %.2f" % fps)
        # Show count
        print("Nombre total de véhicules: %d" % counter_OUT)
        print("Nombre total de véhicules dans la zone jawaz: %d" % counter_JAWAZ)
        print("Nombre total de véhicules dans la zone manuelle: %d" % counter_MANUEL)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #if not FLAGS.dont_show:
            #cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    #cv2.destroyAllWindows()
    # Extract the calculated values and convert them to an Excel file
    # Source from : https://www.programiz.com/python-programming/datetime/current-time
    # https://www.youtube.com/watch?v=7dJsQ4jJ2Wg
    if FLAGS.output_data:
        #now = datetime.now()
        #current_time = now.strftime("%H:%M:%S")
        raw_data = {
            "Nb_IN" : [counter_IN],
            "Nb_OUT" : [counter_OUT],
            "Nb_MANUEL" : [counter_MANUEL],
            "Nb_JAWAZ" : [counter_JAWAZ],
           # "Current_Time" : [current_time]
        }
        df = pd.DataFrame(raw_data, columns=['Nb_IN','Nb_OUT','Nb_MANUEL','Nb_JAWAZ'] )
        print(df)

        writer = pd.ExcelWriter(time.strftime('%y-%m-%d-%H:%M:%S.xlsx'))
        df.to_excel(writer,'Sheet2')
        writer.save()

    # some information on processing single frame
    print("\n-----------------------------------------------------------------------------------------------------------")
    print("[INFO] Estimated time taken to process single frame: {:.4f} seconds".format(elap))
    print("\n[INFO] Estimated total time to finish object detection: {:.4f} minutes".format((elap * frame_num)/60))
    print("------------------------------------------------------------------------------------------------------------")
   
    # Calculate FPS mean()   
    moyenne_fps_frames = moyenne_fps / frame_num  
    print("La moyenne FPS: %.2f" % moyenne_fps_frames)
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
