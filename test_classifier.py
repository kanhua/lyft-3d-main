"""
Test the classifier

"""
from tl_classifier import TLClassifier
import glob
import cv2
import os
import re
import time
from skimage.io import imsave


def decode_state(filename):
    head, tail = os.path.split(filename)
    root, ext = os.path.splitext(tail)
    result = re.match("[A-Za-z0-9]{8}_([\-0-9]+)", root)
    return result.group(1)


data_path = "/Users/kanhua/Dropbox/Programming/udacity-carnd/CarND-Capstone/data/images/*.PNG"
site_data_path="/Users/kanhua/Dropbox/Programming/udacity-carnd/traffic_light_detection/site_dataset/*.jpg"
small_data_path = "dataset/"

output_data_path="/Users/kanhua/Downloads/temp/"

files = glob.glob(site_data_path)

tlc = TLClassifier()
for file in files:
    print(file)
    img_mtx = cv2.imread(file)
    img_mtx = cv2.cvtColor(img_mtx, cv2.COLOR_BGR2RGB)

    t1 = time.time()
    ref_state = tlc.get_classification(img_mtx)
    # ref_state = 0
    result_image = tlc.draw_result(img_mtx)

    _, tail = os.path.split(file)
    basename, ext = os.path.splitext(tail)

    output_file=os.path.join(output_data_path,"{}_result{}".format(basename,ext))

    imsave(output_file,result_image)


    t2 = time.time()
    truth_state = decode_state(file)
    print(ref_state, truth_state, t2 - t1)
