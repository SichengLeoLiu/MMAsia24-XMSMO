import os
import time
import numpy
import argparse
from os.path import join, exists
import glob
import cv2
import numpy as np
from my_train_noise_filter import VideoCoverageLoss
import torch
from operator import itemgetter

def MAP(target, indices, count_frames, k=10):

    #assert logits.shape == target.shape

    #target = target.reshape(-1,k)
   # logits = logits.reshape(-1,k)
    
    #sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    count_nonzero = 0
    map_sum = 0
    for i in range(len(indices)):
        average_precision = 0
        num_rel = 1
        if target[i] == indices[i]:
        #for j in range(indices.shape[1]):
            #if target[i, indices[i, j]] == 1:
            #num_rel += 1
            average_precision += float(num_rel) / count_frames
        #if num_rel==0: continue
        average_precision = average_precision / num_rel
        # print("average_precision: ", average_precision)
        map_sum += average_precision
        count_nonzero += 1
    #return map_sum / indices.shape[0]
    return float(map_sum) / count_nonzero


def precision_at_k(labels, indices, count_frames, k=1, doc_num=10):
    
    #scores = scores.reshape(-1,doc_num) # [batch, doc_num]
    #labels = labels.reshape(-1,doc_num) # [batch, doc_num]

    #sorted, indices = numpy.sort(scores, 1), numpy.argsort(-scores, 1)
    count_nonzero = 0
    precision = 0
    for i in range(len(indices)):
        #num_rel = numpy.sum(labels[i])
        #if num_rel==0: continue
        rel = 0
        for j in range(k):
            if labels[i] == indices[i]:
                rel += 1
        precision += float(rel) / float(k)
        count_nonzero += 1
    return precision / count_nonzero


def main(args):
    dec_pos_dir = join(args.decode_dir, args.decode_pos_folder)
    print("dec_pos_dir", dec_pos_dir)
    dec_dir = join(args.decode_dir, args.decode_folder)
    print("dec_dir", dec_dir)
    #dec_dir = join(args.decode_dir, 'output')
    #with open(join(args.decode_dir, 'log.json')) as f:
    #    split = json.loads(f.read())['split']
    split = 'test'
    ref_dir = join(args.ref_dir, split)
    assert exists(ref_dir)


    dec_pos_files = glob.glob(join(dec_pos_dir, "*.dec"))
    file_id = [os.path.split(x)[1].replace('.dec', '') for x in dec_pos_files]

    decode_pos = []
    decode_pic = []
    ref_pic = []
    count_frame = []
    thumbnail = []
    video = []
    accuracy_list = []
    iou_list = []
    iou1_list = []
    iou2_list = []
    iou3_list = []
    iou4_list = []
    iou5_list = []

    ot_list = []

    counter = 0
    for i in file_id:
        with open(join(dec_pos_dir, '{}.dec'.format(i)),'r') as f:
            decode_pos.append(f.read())

        decode_pic.append(cv2.imread(join(dec_dir, '{}.png'.format(i))))

        
        ref_pic.append(cv2.imread(join(ref_dir, '{}.png'.format(i))))

        
        vidcap = cv2.VideoCapture(join(ref_dir, '{}.mp4'.format(i)))
        success,image = vidcap.read()
        print("join(ref_dir, '{}.mp4'.format(i))", join(ref_dir, '{}.mp4'.format(i)))
        print("success", success)
        print("vidcap", vidcap)
        print("decode_pos[-1]", decode_pos[-1])
        i = 0
        count = 0
        video_frame = []
        while success:
            video_frame.append(image)
            if i % 360:
                count +=1 
            success, image = vidcap.read()
            
            if count == int(decode_pos[-1].replace("tensor([", "").replace("], device='cuda:0')", "")):
                thumbnail.append(image)

            i += 1
            if count > 5000:
                break
        #print("count", count)
        count_frame.append(count)
        video.append(video_frame)
        
        output, output_list = analysis_accuracy(thumbnail, ref_pic)
        accuracy_list.append(output)
        print("accuracy_list", accuracy_list)
        output, output_list, decode_object_list, ref_object_list = analysis_iou(thumbnail, ref_pic)
        iou_list.append(output)
        print("iou_list", iou_list)
        output, output_list, decode_object_list, ref_object_list = analysis_iouk(thumbnail, ref_pic, 1)
        iou1_list.append(output)
        print("iou1_list", iou1_list)

        output, output_list, decode_object_list, ref_object_list = analysis_iouk(thumbnail, ref_pic, 2)
        iou2_list.append(output)

        output, output_list, decode_object_list, ref_object_list = analysis_iouk(thumbnail, ref_pic, 3)
        iou3_list.append(output)

        output, output_list, decode_object_list, ref_object_list = analysis_iouk(thumbnail, ref_pic, 4)
        iou4_list.append(output)

        output, output_list, decode_object_list, ref_object_list = analysis_iouk(thumbnail, ref_pic, 5)
        iou5_list.append(output)

        #output = VideoCoverageLoss(torch.tensor(np.array(thumbnail)), torch.tensor(np.array(video)))
        #ot_list.append(output)
        decode_pos = []
        decode_pic = []
        ref_pic = []
        count_frame = []
        thumbnail = []
        video = []


    print("thumbnail", len(thumbnail))
    #output = MAP(decode_pos, ref_pos, count_frame)
    #metric = 'map'

    metric = 'accuracy'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('accuracy: ' + str(sum(accuracy_list)/len(accuracy_list)))
    print(metric, str(sum(accuracy_list)/len(accuracy_list)))

    #metric = 'iou'
    #with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
    #    f.write('iou: ' + str(sum(iou_list)/len(iou_list)))
    #print(metric, str(sum(iou_list)/len(iou_list)))

    metric = 'iou1'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('iou1: ' + str(sum(iou1_list)/len(iou1_list)))
    print(metric, str(sum(iou1_list)/len(iou1_list)))

    metric = 'iou2'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('iou2: ' + str(sum(iou2_list)/len(iou2_list)))
    print(metric, str(sum(iou2_list)/len(iou2_list)))

    metric = 'iou3'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('iou3: ' + str(sum(iou3_list)/len(iou3_list)))
    print(metric, str(sum(iou3_list)/len(iou3_list)))

    metric = 'iou4'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('iou: ' + str(sum(iou4_list)/len(iou4_list)))
    print(metric, str(sum(iou4_list)/len(iou4_list)))

    metric = 'iou5'
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write('iou5: ' + str(sum(iou5_list)/len(iou5_list)))
    print(metric, str(sum(iou5_list)/len(iou5_list)))
          
   # metric = 'visual_ot'
    #with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
    #    f.write('visual_ot: ' + str(sum(ot_list)/len(ot_list)))
    #print(metric, str(sum(ot_list)/len(ot_list)))
    #print(metric, output)
    #with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
   #     f.write(str(output))


    #output = precision_at_k(decode_pos, ref_pos, count_frame)
    #metric = 'precision_at_k'

    #print(metric, output)
    #with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
    #    f.write(str(output))

    
        


def calculateEudDistance(i1, i2): #lower-level visual features 
    #distance = 0
    #for i in range(len(i1)):
     #   distance += np.square(i1[i]-i2[i])
    #print("i1", i1.shape)
    #print("i2", i2.shape)
    #print("i1amax", np.amax(i1))
    #print("i1amin", np.amin(i1))
    #print("i2amax", np.amax(i2))
   # print("i2amin", np.amin(i2))
    #distance = np.sqrt(np.sum((i1-i2)**2))
    distance = np.linalg.norm(i1-i2)
    #print("eud", distance)
    #return np.sqrt(distance)
    #print("norm", np.ones(( 360, 640))*256)
    #print("norm dis", np.sqrt(np.sum((np.ones(( 360,640,3))*255)**2)))
    norm_dis = np.linalg.norm(np.ones(( 360,640,3))*255)
    #print("norm_dis", norm_dis)
    distance = distance / norm_dis
    #distance = np.linalg.norm(i1-i2)/np.linalg.norm((np.ones(( 360, 640))*256-np.zeros((360, 640))))
    #print("eud2", distance)
    return distance


def analysis_accuracy(decode_list, ground_list): #DeepQAMVS: Query-Aware Hierarchical Pointer Networks for Multi-Video Summarization
    correct = 0
    distance = []
    #print("decode_list", decode_list)
    for i1, i2 in zip(decode_list, ground_list):
        i2_resize = cv2.resize(i2, (640, 360))
        i1_resize = cv2.resize(i1, (640, 360))
        #i2_norm = cv2.normalize(i2_resize, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
        #i1_norm = cv2.normalize(i1_resize, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
        dis = calculateEudDistance(i1_resize, i2_resize)
        distance.append(dis)
        #if calculateEudDistance(i1_norm, i2_norm) <= 0.6:
        if dis <= 0.6:
            correct += 1
    #print("correct", correct)
    #print("decode_list", decode_list)
    acc = correct/len(decode_list)
    #print("acc", acc, "distance", distance)
    return acc, distance

def get_objects(image, net):
    CONF = 0.5
    THRESHOLD = 0.5
    # load our input image and grab its spatial dimensions
    labelsPath = os.path.sep.join([args.yolo_folder, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    #image = cv2.imread(image)
    #(H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    #print("ln", ln)
    #print("net.getUnconnectedOutLayers()", net.getUnconnectedOutLayers())
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
    # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONF:
                #box = detection[0:4] * np.array([W, H, W, H])
                confidences.append(float(confidence))
                classIDs.append(classID)
    #idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF, THRESHOLD)

    detected_labels = []

    #for i in idxs.flatten():
    for i in classIDs:
        #detected_labels.append(LABELS[classIDs[i]])
        detected_labels.append(LABELS[i])

    return detected_labels, confidences

def get_iou(item1,item2):
    

    intersection = 0
    for item in item1:
        if item in item2:
            intersection+=1
    union = len(item1)+len(item2)-intersection

    if union > 0:
        return intersection*1.0 / union
    else:
        return 1

def analysis_iou(decode_list, ground_list): #Deep Reinforcement Learning for Query-Conditioned Video Summarization
    #higher-level semantic information
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args.yolo_folder, "yolov3.weights"])
    configPath = os.path.sep.join([args.yolo_folder, "yolov3.cfg"])
    # load our YOLO object detector trained on COCO dataset (80 classes)
    #print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    #print("net", net)

    iou_list = []
    i1_object_list = []
    i2_object_list = []
    for i1, i2 in zip(decode_list, ground_list):
        i1_object, i1_conf = get_objects(i1, net)
        i2_object, i2_conf = get_objects(i2, net)
        i1_object_list.append(','.join(i1_object))
        i2_object_list.append(','.join(i2_object))
       # print("i1_object", i1_object)
        #print("i2_object", i2_object)
        iou_list.append(get_iou(i1_object,i2_object))

    return sum(iou_list)/len(iou_list), iou_list, i1_object_list, i2_object_list


def analysis_iouk(decode_list, ground_list, k): #Deep Reinforcement Learning for Query-Conditioned Video Summarization
    #higher-level semantic information
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args.yolo_folder, "yolov3.weights"])
    configPath = os.path.sep.join([args.yolo_folder, "yolov3.cfg"])
    # load our YOLO object detector trained on COCO dataset (80 classes)
    #print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    #print("net", net)

    iou_list = []
    i1_object_list = []
    i2_object_list = []
    for i1, i2 in zip(decode_list, ground_list):
        i1_object, i1_conf = get_objects(i1, net)
        i2_object, i2_conf = get_objects(i2, net)

        i1_obj_conf = dict(zip(i1_object, i1_conf))
        i2_obj_conf = dict(zip(i2_object, i2_conf))
        i1_obj_conf = dict(sorted(i1_obj_conf.items(), key = itemgetter(1), reverse = True)[:k]).keys()
        i2_obj_conf = dict(sorted(i2_obj_conf.items(), key = itemgetter(1), reverse = True)[:k]).keys()

        i1_object_list.append(','.join(i1_obj_conf))
        i2_object_list.append(','.join(i2_obj_conf))
       # print("i1_object", i1_object)
        #print("i2_object", i2_object)
        print("i1_object_list", i1_object_list)
        print("i2_object_list", i2_object_list)
        iou_list.append(get_iou(i1_object,i2_object))

    return sum(iou_list)/len(iou_list), iou_list, i1_object_list, i2_object_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the output files for the RL full models')

    parser.add_argument('--ref_dir', action='store', required=True,
                        help='directory of ref summaries')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--decode_folder', action='store', required=True,
                        help='folder of decoded summaries')
    parser.add_argument('--decode_pos_folder', action='store', required=True,
                        help='folder of decoded summaries')
    parser.add_argument('--yolo_folder', action='store', required=True,
                        help='yolo folder')
    args = parser.parse_args()
    main(args)

