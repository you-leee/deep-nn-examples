import os
from keras import backend as K
from keras.models import load_model
from ConvolutionalNeuralNetworks.yolo.yolo_funcs import yolo_eval, read_classes, read_anchors, yolo_head, predict

if __name__ == '__main__':
    sess = K.get_session()

    class_names = read_classes("yolo/model/coco_classes.txt")
    anchors = read_anchors("yolo/model/yolo_anchors.txt")
    image_shape = (608., 608.)

    yolo_model = load_model("yolo/model/yolo.h5")
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

    image_path = "../datasets/yolo_images/"
    for file in os.listdir(image_path):
        if os.path.isfile(os.path.join(image_path, file)):
            predict(sess, yolo_model,image_path, file, scores, boxes, classes, class_names)
