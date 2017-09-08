import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

import caffe
from caffe.proto import caffe_pb2

def main(args, meanarr):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # and the image you would like to classify.
    MODEL_FILE = args["deploy"]
    PRETRAINED = args["model"]

    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=meanarr.mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

    image_input = caffe.io.load_image(args["image"])
    predict = net.predict([image_input])
    print(predict)

    print("--------")

    print('prediction shape: %s' % predict[0].shape)
    print('predicted class: %s' % predict[0].argmax())

    proba = predict[0][predict[0].argmax()]
    ind = predict[0].argsort()[-2:][::-1] # top-5 predictions
    print("proba: %s --  ind: %s" % (proba , ind))



if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Test images")
    args.add_argument("--deploy")
    args.add_argument("--model")
    args.add_argument("--image")
    args.add_argument("--mean")
    parsed = vars(args.parse_args())

    mean_blob = caffe_pb2.BlobProto()
    with open(parsed["mean"], "rb") as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

    #print(parsed["mean"], mean_array)

    main(parsed, mean_array)
