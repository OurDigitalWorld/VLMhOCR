"""
accEst.py - get average accuracy from HOCR files

Usage (see list of options):
    accEst.py [-h]

For example:
    accEst.py -f gk -e JPG

Simple averaging. Look for image files and
check for corresponding HOCR files, not the other
way around. Tesseract sometimes hits a wall
with recognition.

- art rhyno, u. of windsor & ourdigitalworld
"""

import argparse, glob, math, os, sys, time, tempfile
import xml.etree.ElementTree as ET

HOCR_NS = 'http://www.w3.org/1999/xhtml' #namespace for HOCR


""" pull conf from bbox string """
def getBBoxConf(bbox_str):
    conf = None

    if ';' in bbox_str:
        bbox_info = bbox_str.split(';')
        bbox_info = bbox_info[1].strip()
        bbox_info = bbox_info.split(' ')
        conf = int(bbox_info[1])
    bbox_info = bbox_str.replace(';',' ')
    bbox_info = bbox_info.split(' ')

    return int(conf)

""" step thru HOCR """
def sortOutHocrAcc(tree):
    word_cnt = 0
    conf_sum = 0

    for span_elem in tree.iterfind('.//{%s}%s' % (HOCR_NS,'span')):
        class_name = span_elem.attrib['class']
        if class_name == 'ocrx_word' and span_elem.text is not None: 
            conf = getBBoxConf(span_elem.attrib['title'])
            if conf >= 0:
                word_cnt += 1
                conf_sum += conf
    return word_cnt, conf_sum

#parser values
parser = argparse.ArgumentParser()
arg_named = parser.add_argument_group("named arguments")
arg_named.add_argument('-e', '--ext', type=str,
    default="jpg",
    help="extension of image format, e.g. tiff")
arg_named.add_argument("-f","--folder",
    help="input folder (contains image & hocr files)")

args = parser.parse_args()

if args.folder == None or not os.path.exists(args.folder):
    print("missing image folder, use '-h' parameter for syntax")
    sys.exit()

sum_words = 0
sum_acc = 0
for img in sorted(glob.glob(args.folder + "/*." + args.ext)):
    hocr_file = img.replace('.' + args.ext,'.hocr')

    if not os.path.exists(hocr_file):
        print("%s not detected" % hocr_file)
    else:
        print("sort through hocr words for " + hocr_file)
        try:
            tree = ET.ElementTree(file=hocr_file)
        except:
            print("HOCR parse problem for %s" % hocr_file)
            tree = None
        if tree is not None:
            word_cnt,conf_sum = sortOutHocrAcc(tree)
            sum_words += word_cnt
            sum_acc += conf_sum
            avg_acc = round(conf_sum/word_cnt,2)
            print("word cnt: %d, acc sum: %d avg: %.2f" % 
                  (word_cnt,conf_sum,avg_acc))

avg_acc = round(sum_acc/sum_words,2)
print("Final word cnt: %d, acc sum: %d, acc avg: %.2f" % (sum_words,sum_acc, avg_acc))
