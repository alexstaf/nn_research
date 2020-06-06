import numpy as np
from scipy.optimize import linear_sum_assignment

import cv2
import tensorflow as tf


FONT = cv2.FONT_HERSHEY_SIMPLEX

CLASSES = {
    404: 'vehicle',
    103: 'land vehicle',
    40: 'bicycle',
    571: 'car',
    300: 'motorcycle',
    50: 'bus',
    393: 'train',
    400: 'truck',
    106: 'watercraft',
    43: 'boat',
    572: 'aircraft',
    468: 'airplane',
    424: 'helicopter',
    306: 'missile',
    69: 'person',
    308: 'man',             # remapped to Person
    228: 'woman',           # remapped to Person
    408: 'weapon',
    384: 'tank'
}

CLASSES_KEYS = list(CLASSES.keys())
CLASSES_VALUES = list(CLASSES.values())


def crop(image):
    if image.shape[0] == image.shape[1]:
        return image
    y = image.shape[0] // 2
    x = image.shape[1] // 2
    if y > x:
        image = image[y - x:y + x, :, :]
    else:
        image = image[:, x - y:x + y, :]
    return image


def extend(image):
    if image.shape[0] == image.shape[1]:
        return image
    y = image.shape[0]
    x = image.shape[1]
    if y > x:
        im = np.zeros((y, y, 3), np.uint8) + 128
        im[:, (y - x) // 2:(y - x) // 2 + x, :] = image
    else:
        im = np.zeros((x, x, 3), np.uint8) + 128
        im[(x - y) // 2:(x - y) // 2 + y, :, :] = image
    return im


def clip_box(box, shape):
    """
    Clip box for given image shape.

    Args:
        box (array_like[int]): Box for clipping in the next format:
            [y_min, x_min, y_max, x_max].
        shape (tuple[int]): Shape of image.

    Returns:
        array_like[int]: Clipped box.

    """
    ymin, xmin, ymax, xmax = box
    if ymin < 0:
        ymin = 0
    elif ymin >= shape[0]:
        ymin = shape[0] - 1
    box[0] = ymin
    if xmin < 0:
        xmin = 0
    elif xmin >= shape[1]:
        xmin = shape[1] - 1
    box[1] = xmin
    if ymax < 0:
        ymax = 0
    elif ymax >= shape[0]:
        ymax = shape[0] - 1
    box[2] = ymax
    if xmax < 0:
        xmax = 0
    elif xmax >= shape[1]:
        xmax = shape[1] - 1
    box[3] = xmax
    return box


def calc_IoU(bbox1, bbox2):
    """
    Calculate intersection over union for two boxes.

    Args:
        bbox1 (array_like[int]): Endpoints of the first bounding box
            in the next format: [ymin, xmin, ymax, xmax].
        bbox2 (array_like[int]): Endpoints of the second bounding box
            in the next format: [ymin, xmin, ymax, xmax].

    Returns:
        float: Intersection over union for given bounding boxes.

    """
    ymin1, xmin1, ymax1, xmax1 = bbox1
    ymin2, xmin2, ymax2, xmax2 = bbox2
    ymin = max(ymin1, ymin2)
    xmin = max(xmin1, xmin2)
    ymax = min(ymax1, ymax2)
    xmax = min(xmax1, xmax2)
    if xmax <= xmin or ymax <= ymin:
        return 0
    intersection = (ymax - ymin) * (xmax - xmin)
    union = ((ymax1 - ymin1) * (xmax1 - xmin1) +
             (ymax2 - ymin2) * (xmax2 - xmin2) -
             intersection)
    return intersection / union


def make_pairs(prev_boxes, boxes):
    """
    Give correspondence for bounding boxes from two lists.

    Args:
        prev_boxes (list[array_like[int]]): Previous bounding boxes
            in the next format: [box_1, box_2, ...],
            where box_* is [y_min, x_min, y_max, x_max].
        boxes (list[array_like[int]]): Next bounding boxes
            in the next format: [box_1, box_2, ...],
            where box_* is [y_min, x_min, y_max, x_max].

    Returns:
        tuple[array[int]]: Correspondence of indices of given lists.

    """
    # Cost matrix construction.
    cost_mat = []
    for i in range(len(prev_boxes)):
        cost_mat.append([])
        for j in range(len(boxes)):
            cost_mat[i].append(1 - calc_IoU(prev_boxes[i], boxes[j]))

    # Use the Hungarian algorithm.
    return linear_sum_assignment(np.array(cost_mat, ndmin=2))


def print_info_on_frame(frame, bboxes, classes, scores):
    """
    Print information on the given frame.

    Args:
        frame (numpy.ndarray[int]): Frame to print information.
        bboxes (list[array_like[int]]): Endpoints of a bounding boxes
            in the next format: [box_1, box_2, ...],
            where box_* is [y_min, x_min, y_max, x_max].
        classes (list[int]): Detection classes.
        scores (list[float]): Detection scores.
    """
    for i in range(len(bboxes)):
        if bboxes[i] is not None:
            ymin, xmin, ymax, xmax = bboxes[i]
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            box = [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = clip_box(box, frame.shape)
            if ymax - ymin != 0 and xmax - xmin != 0:
                if isinstance(classes[i], str):
                    k = 0
                else:
                    k = ((CLASSES_KEYS.index(classes[i]) + 1) /
                         (len(CLASSES_KEYS) + 1))
                color = cv2.cvtColor(
                    np.array([[[k * 180, 255, 255]]], dtype=np.uint8),
                    cv2.COLOR_HSV2BGR
                )
                color = tuple([int(x) for x in color[0, 0]])
                frame[ymin:ymax, xmin] = color
                frame[ymin:ymax, xmax] = color
                frame[ymin, xmin:xmax] = color
                frame[ymax, xmin:xmax] = color
                if isinstance(classes[i], str):
                    text = classes[i]
                else:
                    text = CLASSES[classes[i]]
                text = '{}: {}%'.format(text, int(scores[i] * 100))
                cv2.putText(frame, text, (xmin, ymin + 25), FONT, 1, color)


def filter_output_dict(output_dict):
    assert (
        len(output_dict['detection_boxes']) ==
        len(output_dict['detection_scores']) and
        len(output_dict['detection_classes']) ==
        len(output_dict['detection_scores'])
    )
    output_dict['detection_boxes'] = list(output_dict['detection_boxes'])
    output_dict['detection_scores'] = list(output_dict['detection_scores'])
    output_dict['detection_classes'] = list(output_dict['detection_classes'])
    for k in range(len(output_dict['detection_scores'])):
        try:
            sep = list(output_dict['detection_scores'][k]).index(0.)
            output_dict['detection_boxes'][k] = list(
                output_dict['detection_boxes'][k][:sep]
            )
            output_dict['detection_classes'][k] = list((
                output_dict['detection_classes'][k][:sep]
            ).astype(int))
            output_dict['detection_scores'][k] = list(
                output_dict['detection_scores'][k][:sep]
            )
        except ValueError as e:
            if str(e) != '0.0 is not in list':
                raise
        i = 0
        while i < len(output_dict['detection_classes'][k]):
            if output_dict['detection_classes'][k][i] not in CLASSES_KEYS:
                del output_dict['detection_boxes'][k][i]
                del output_dict['detection_scores'][k][i]
                del output_dict['detection_classes'][k][i]
            else:
                i += 1


class Network:
    def __init__(self, ckpt):
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def __del__(self):
        self.release()

    def run_inference_for_images(self, images):
        """
        Run inference for images.

        Args:
            images (numpy.ndarray[int]): Images for inference
                in the next format: [image_1, image_2, ...],
                where image_* is image with same shape.

        Returns:
            dict[str, ...]: Output dictionary with detections.

        """
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes',
            'detection_scores', 'detection_classes'
        ]:
            tensor_name = key + ':0'
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name
            )
        image_tensor = tf.get_default_graph().get_tensor_by_name(
            'image_tensor:0'
        )
        output_dict = self.sess.run(tensor_dict,
                                    feed_dict={image_tensor: images})
        return output_dict

    def release(self):
        self.sess.close()
