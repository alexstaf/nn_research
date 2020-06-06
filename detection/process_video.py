import os
import argparse
import numpy as np

import cv2
import json
import tensorflow as tf

from common import (Network, filter_output_dict, print_info_on_frame, calc_IoU,
                    make_pairs)


WINDOW_NAME = 'detections'
TRACKER_CREATE = cv2.TrackerMOSSE_create


def load_graph(frozen_graph_filename):
    """Load frozen graph.

    Args:
        frozen_graph_filename (str): Path to file with frozen graph.

    Returns:
        tf.Graph: Graph imported from given file.

    """
    with tf.io.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='import')
        return graph


def bbox_to_rect(bbox, shape):
    ymin, xmin, ymax, xmax = bbox
    ymin = int(ymin * shape[0])
    ymax = int(ymax * shape[0])
    xmin = int(xmin * shape[1])
    xmax = int(xmax * shape[1])
    return xmin, ymin, xmax - xmin, ymax - ymin


def rect_to_bbox(rect, shape):
    x, y, w, h = rect
    return np.array([y / shape[0], x / shape[1],
                     (y + h) / shape[0], (x + w) / shape[1]])


def remap_classes(classes):
    for i in range(len(classes)):
        if classes[i] == 308 or classes[i] == 228:
            classes[i] = 69


def nms(boxes, scores, classes, c=0.5):
    indices = []
    length = len(boxes)
    for i in range(length):
        disgard = False
        for j in range(length):
            if i != j and calc_IoU(boxes[i], boxes[j]) > c:
                if scores[j] > scores[i]:
                    disgard = True
                    break
        if not disgard:
            indices.append(i)
    return indices


def do_nms(boxes, scores, classes, indices):
    _boxes = []
    _scores = []
    _classes = []
    for i in indices:
        _boxes.append(boxes[i])
        _scores.append(scores[i])
        _classes.append(classes[i])
    return _boxes, _scores, _classes


def filter_detections_by_class(output_dict, cl):
    detection_classes = output_dict['detection_classes'][0]
    indices = []
    for i in range(len(detection_classes)):
        det_cl = detection_classes[i]
        if det_cl == cl:
            indices.append(i)
    return indices


def update_tracker(trackers, output_dict, frame, i=-1):
    ret, rect = trackers[i].update(frame)
    if ret:
        output_dict['detection_boxes'][0][i] = rect_to_bbox(
            rect, frame.shape
        )
    else:
        del trackers[i]
        del output_dict['detection_boxes'][0][i]
        del output_dict['detection_scores'][0][i]
        del output_dict['detection_classes'][0][i]


def parse_args():
    """Return parsed arguments from command line or file.

    Returns:
        Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description='Detect objects on video.', fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-v', '--video', required=True,
                        help='path to video')
    parser.add_argument('-c', '--ckpt',
                        default='../data/'
                        'faster_rcnn_inception_resnet_v2_atrous_oi4.pb',
                        help='checkpoint for detection')
    parser.add_argument('-n', '--num-frames', type=int, default=5,
                        help='path to video')
    parser.add_argument('-d', '--data', default='../data',
                        help='path to model data')
    parser.add_argument('-s', '--save', help='path to save video (optional)')
    return parser.parse_args()


def main():
    # Parse arguments from command line or file.
    args = parse_args()
    try:
        args.video = int(args.video)
    except ValueError:
        pass

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()

    graph = tf.Graph()
    with graph.as_default():
        network = Network(args.ckpt)

    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with open(os.path.join(args.data, 'classes.json')) as f:
        classes = json.load(f)

    graph_224 = load_graph(os.path.join(args.data, 'inception_v1_224.pb'))
    sess_224 = tf.compat.v1.Session(graph=graph_224, config=session_config)
    x_224 = graph_224.get_tensor_by_name('import/input:0')
    y_224 = graph_224.get_tensor_by_name('import/ArgMax:0')

    graph_64 = load_graph(os.path.join(args.data, 'inception_v1_64.pb'))
    sess_64 = tf.compat.v1.Session(graph=graph_64, config=session_config)
    x_64 = graph_64.get_tensor_by_name('import/input:0')
    y_64 = graph_64.get_tensor_by_name('import/ArgMax:0')

    graph_32 = load_graph(os.path.join(args.data, 'densenet_32.pb'))
    sess_32 = tf.compat.v1.Session(graph=graph_32, config=session_config)
    x_32 = graph_32.get_tensor_by_name('import/input:0')
    y_32 = graph_32.get_tensor_by_name('import/ArgMax:0')

    def classify_object(frame, bbox):
        image = frame[int(frame.shape[0] * bbox[0]):
                      int(frame.shape[0] * bbox[2]),
                      int(frame.shape[1] * bbox[1]):
                      int(frame.shape[1] * bbox[3])]
        c_224, c_64, c_32 = None, None, None
        if image.shape[0] >= 224 and image.shape[1] >= 224:
            image_224 = cv2.resize(image, (224, 224))
            with sess_224.as_default():
                c_224 = sess_224.run(y_224, feed_dict={x_224: [image_224]})[0]
                cl = classes[c_224]
        elif image.shape[0] >= 64 and image.shape[1] >= 64:
            image_64 = cv2.resize(image, (64, 64))
            with sess_64.as_default():
                c_64 = sess_64.run(y_64, feed_dict={x_64: [image_64]})[0]
                cl = classes[c_64]
        else:
            image_32 = cv2.resize(image, (32, 32))
            with sess_32.as_default():
                c_32 = sess_32.run(y_32, feed_dict={x_32: [image_32]})[0]
                cl = classes[c_32]
        return cl

    ret, frame = cap.read()

    writer = None
    if ret and args.save:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        height, width, _ = frame.shape
        writer = cv2.VideoWriter(args.save, fourcc, 30., (width, height))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    k = 0
    classes_list = [404, 103, 106, 572]
    output_dict = None
    trackers = None
    while ret:
        if k % args.num_frames == 0:
            output_dict_prev = output_dict
            with graph.as_default():
                output_dict = network.run_inference_for_images(
                    np.asarray([frame[..., ::-1]])
                )
            filter_output_dict(output_dict)
            remap_classes(output_dict['detection_classes'][0])
            nms_indices = nms(output_dict['detection_boxes'][0],
                              output_dict['detection_scores'][0],
                              output_dict['detection_classes'][0])
            (output_dict['detection_boxes'][0],
             output_dict['detection_scores'][0],
             output_dict['detection_classes'][0]) = do_nms(
                output_dict['detection_boxes'][0],
                output_dict['detection_scores'][0],
                output_dict['detection_classes'][0],
                nms_indices
            )
            for i in range(len(output_dict['detection_boxes'][0])):
                bbox, det_class = (output_dict['detection_boxes'][0][i],
                                   output_dict['detection_classes'][0][i])
                if det_class in classes_list:
                    obj_class = classify_object(frame[..., ::-1], bbox)
                    output_dict['detection_classes'][0][i] = obj_class
            if output_dict_prev is not None:
                output_dict_next = {
                    'detection_boxes': [[]],
                    'detection_scores': [[]],
                    'detection_classes': [[]]
                }
                trackers_next = []
                classes_keys = []
                for key in output_dict_prev['detection_classes'][0]:
                    if key not in classes_keys:
                        classes_keys.append(key)
                for key in output_dict['detection_classes'][0]:
                    if key not in classes_keys:
                        classes_keys.append(key)
                for key in classes_keys:
                    indices_prev = filter_detections_by_class(output_dict_prev,
                                                              key)
                    det_prev = [[], [], []]
                    for i in indices_prev:
                        det_prev[0].append(
                            output_dict_prev['detection_boxes'][0][i]
                        )
                        det_prev[1].append(
                            output_dict_prev['detection_scores'][0][i]
                        )
                        det_prev[2].append(
                            output_dict_prev['detection_classes'][0][i]
                        )
                    indices = filter_detections_by_class(output_dict, key)
                    det = [[], [], []]
                    for i in indices:
                        det[0].append(
                            output_dict['detection_boxes'][0][i]
                        )
                        det[1].append(
                            output_dict['detection_scores'][0][i]
                        )
                        det[2].append(
                            output_dict['detection_classes'][0][i]
                        )
                    output_dict_next['detection_boxes'][0].extend(det[0])
                    output_dict_next['detection_scores'][0].extend(det[1])
                    output_dict_next['detection_classes'][0].extend(det[2])
                    for bbox in det[0]:
                        trackers_next.append(TRACKER_CREATE())
                        trackers_next[-1].init(frame,
                                               bbox_to_rect(bbox, frame.shape))
                    row_ind, col_ind = make_pairs(det_prev[0], det[0])
                    for i in range(len(det_prev[0])):
                        if i not in row_ind:
                            output_dict_next['detection_boxes'][0].append(
                                det_prev[0][i]
                            )
                            output_dict_next['detection_scores'][0].append(
                                det_prev[1][i]
                            )
                            output_dict_next['detection_classes'][0].append(
                                det_prev[2][i]
                            )
                            trackers_next.append(trackers[indices_prev[i]])
                            update_tracker(trackers_next, output_dict_next,
                                           frame)
                    for i in range(len(row_ind)):
                        if calc_IoU(det_prev[0][row_ind[i]],
                                    det[0][col_ind[i]]) == 0:
                            output_dict_next['detection_boxes'][0].append(
                                det_prev[0][row_ind[i]]
                            )
                            output_dict_next['detection_scores'][0].append(
                                det_prev[1][row_ind[i]]
                            )
                            output_dict_next['detection_classes'][0].append(
                                det_prev[2][row_ind[i]]
                            )
                            trackers_next.append(trackers[indices_prev[i]])
                            update_tracker(trackers_next, output_dict_next,
                                           frame)
                output_dict = output_dict_next
                nms_indices = nms(output_dict['detection_boxes'][0],
                                  output_dict['detection_scores'][0],
                                  output_dict['detection_classes'][0])
                (output_dict['detection_boxes'][0],
                 output_dict['detection_scores'][0],
                 output_dict['detection_classes'][0]) = do_nms(
                    output_dict['detection_boxes'][0],
                    output_dict['detection_scores'][0],
                    output_dict['detection_classes'][0],
                    nms_indices
                )
                trackers = []
                for i in nms_indices:
                    trackers.append(trackers_next[i])
            else:
                trackers = []
                for bbox in output_dict['detection_boxes'][0]:
                    trackers.append(TRACKER_CREATE())
                    trackers[-1].init(frame, bbox_to_rect(bbox, frame.shape))
        else:
            i = 0
            while i < len(output_dict['detection_boxes'][0]):
                update_tracker(trackers, output_dict, frame, i)
                i += 1
        print_info_on_frame(frame, output_dict['detection_boxes'][0],
                            output_dict['detection_classes'][0],
                            output_dict['detection_scores'][0])
        if writer:
            writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(33)
        if key == ord('q') or key == 27:
            break
        ret, frame = cap.read()
        k += 1
    cv2.destroyWindow(WINDOW_NAME)
    cap.release()
    if writer:
        writer.release()


if __name__ == '__main__':
    main()
