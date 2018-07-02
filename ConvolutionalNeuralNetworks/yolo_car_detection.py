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

    predict(sess,yolo_model, "1.jpg", scores, boxes, classes, class_names)
    predict(sess,yolo_model, "2.jpg", scores, boxes, classes, class_names)
    predict(sess,yolo_model, "3.jpg", scores, boxes, classes, class_names)
    predict(sess,yolo_model, "4.jpg", scores, boxes, classes, class_names)
    predict(sess,yolo_model, "5.jpg", scores, boxes, classes, class_names)
    predict(sess,yolo_model, "6.jpg", scores, boxes, classes, class_names)
