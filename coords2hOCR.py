"""
coords2hOCR.py - create hOCR file using coordinates

Usage (see list of options):
    coords2hOCR.py [-h] 

For example:
    coords2hOCR.py -b sources/comber.jpg
    coords2hOCR.py -b sources/comber.jpg -f regions -t

By default, this script combines multiple hOCR files
into one. The coordinates are recalculated based on
the file name, for example:

    imgs/para_00372_02426_01558_02923.hocr

The last 4 numbers (372,2426,1558,2923) indicate
the coordinates from the original image. Looping
through the hOCR files from the images produced 
by eyPage.py, one hOCR file is produced. The
t flag is used for text files, with entries like:

    sale(824,20),(968,121)

- art rhyno, u. of windsor & ourdigitalworld
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse, glob, math, os, sys
import copy,cv2

#namespace for HOCR
HOCR_NS = 'http://www.w3.org/1999/xhtml'
ET.register_namespace('html', HOCR_NS)
MARGIN = 10
HNUM = 1000
OCR_OPTIONS = 'ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf'

""" hocr_region - a rectangle on the image """
class hocr_region:
    def __init__(self, text, line, x0, y0, x1, y1, conf):
        self.text = text
        self.line = line
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.conf = conf

""" add headers for HOCR """
def addHtmlHeaders(img_base):
    html_node = ET.Element(ET.QName(HOCR_NS,"html"))
    head_element = ET.Element(ET.QName(HOCR_NS,"head"))
    title_element = ET.Element(ET.QName(HOCR_NS,"title"))
    title_element.text = img_base
    head_element.append(title_element)
    meta_element = ET.Element(ET.QName(HOCR_NS,"meta"))
    meta_element.set('http-equiv','Content-Type')
    meta_element.set('content','text/html;charset=utf-8')
    head_element.append(meta_element)
    meta_element = ET.Element(ET.QName(HOCR_NS,"meta"))
    meta_element.set('ocr-system','eynollah+hunyuanOCR')
    head_element.append(meta_element)
    meta_element = ET.Element(ET.QName(HOCR_NS,"meta"))
    meta_element.set('ocr-capabilities',OCR_OPTIONS)
    head_element.append(meta_element)
    html_node.append(head_element)

    return html_node

""" write hocr file """
def writeHocr(node,hocr_file):

    #use minidom pretty print feature
    xmlstr = minidom.parseString(
                 ET.tostring(node)).toprettyxml(indent="   ")
    with open(hocr_file, 'w') as f:
        f.write(xmlstr)
    f.close()

""" sort out coords based on child elements """
def getBBox(parent):

    low_x = 0
    low_y = 0
    high_x = 0
    high_y = 0

    #for child in parent.getchildren():
    for child in list(parent):
        x0,y0,x1,y1,_ = getBBoxInfo(child.attrib['title'])
        if x0 < low_x or low_x == 0:
            low_x = x0
        if y0 < low_y or low_y == 0:
            low_y = y0
        if x1 > high_x:
            high_x = x1
        if y1 > high_y:
            high_y = y1

    return "bbox %d %d %d %d" % (low_x,low_y,high_x,high_y)

""" get starting x and y """
def getBase(HOCRfile):
    parts = HOCRfile.split('_')
    base_x = int(parts[1])
    base_y = int(parts[2])

    return base_x, base_y

""" pull coords and sometimes conf from bbox string """
def getBBoxInfo(bbox_str,adj=False):
    margin = 0
    if adj: margin = MARGIN
    conf = None

    if ';' in bbox_str:
        bbox_info = bbox_str.split(';')
        bbox_info = bbox_info[1].strip()
        bbox_info = bbox_info.split(' ')
        conf = int(bbox_info[1])
    bbox_info = bbox_str.replace(';',' ')
    bbox_info = bbox_info.split(' ')
    x0 = int(bbox_info[1]) - margin
    y0 = int(bbox_info[2]) - margin
    x1 = int(bbox_info[3]) - margin
    y1 = int(bbox_info[4]) - margin

    return x0,y0,x1,y1,conf

""" is coordinate in range """
def outOfRange(y0,last_y):
    if y0 < (last_y + MARGIN) and y0 > (last_y - MARGIN):
        return True
    return False

""" pull out coordinates from text description """
def runThruText(TEXTfile,adj_w,adj_h,wregions):
    base_x, base_y = getBase(TEXTfile)
    line_id = None

    last_y = y0 = 0
    with open(TEXTfile, "r") as file:
        for line in file:
            text = line.strip()
            lidx = text.rfind('(')
            idx = text.rfind('(', 0, lidx)
            coords = text[idx:]
            text = text[:idx]
            if len(text) > 0 and text != '.':
                coords = coords.replace('(','').replace(')','')
                points = coords.split(',')
                last_y = y0
                if len(points) == 4:
                    x0 = math.floor((int(points[0]))*adj_w)
                    y0 = math.floor((int(points[1]))*adj_h)
                    x1 = math.floor((int(points[2]))*adj_w)
                    y1 = math.floor((int(points[3]))*adj_h)
                    if line_id is None or outOfRange(y0,last_y):
                        line_id = '%s_%d_%d_%d_%d' % (TEXTfile,x0,y0,x1,y1)

                    wregions.append(hocr_region(text,line_id,
                        base_x + x0 - MARGIN,
                        base_y + y0 - MARGIN,
                        base_x + x1 - MARGIN,
                        base_y + y1 - MARGIN,-1))

""" pull together paragraphs from hocr file """
def runThruHocr(HOCRfile,HOCRconf,wregions):

    base_x, base_y = getBase(HOCRfile)
    tree = ET.ElementTree(file=HOCRfile)

    for elem in tree.iterfind('.//{%s}%s' % (HOCR_NS,'p')):
        line_info = None
        if 'class' in elem.attrib:
            class_name = elem.attrib['class']
            if class_name == 'ocr_par': 
                words = ''
                for word_elem in elem.iterfind('.//{%s}%s' % (HOCR_NS,'span')):
                    class_name = word_elem.attrib['class']
                    if class_name == 'ocr_line': #save line infos
                        line_id = HOCRfile + '_' + word_elem.attrib['id']
                    if class_name == 'ocrx_word': #word details
                        word_text = word_elem.text.strip()
                        if len(word_text) > 0:
                            x0,y0,x1,y1,conf = getBBoxInfo(
                                word_elem.attrib['title'],True)
                            if conf >= HOCRconf:
                                wregions.append(hocr_region(
                                    word_text,
                                    line_id,
                                    base_x + x0,
                                    base_y + y0,
                                    base_x + x1,
                                    base_y + y1,
                                    conf))

""" step through hOCR format """
def sortOutWords(wregions,w,h,HOCRfile,HOCRin):
    #hocr numbering starts at 1 
    page_cnt = 1
    block_cnt = 1
    par_cnt = 1
    line_cnt = 1
    word_cnt = 1

    parent_node = addHtmlHeaders(HOCRin)
    body_node = ET.Element(ET.QName(HOCR_NS,"body"))
    pg_element = ET.Element(ET.QName(HOCR_NS,"div"))
    pg_element.set('class','ocr_page')
    pg_element.set('title','\"%s\"; bbox 0 0 %d %d; ppageno 0' % (HOCRin,w,h))
    block_element = ET.Element(ET.QName(HOCR_NS,"div"))
    block_element.set('class','ocr_carea')

    last_line = None
    last_para = None
    l_element = ET.Element(ET.QName(HOCR_NS,"span"))
    p_element = ET.Element(ET.QName(HOCR_NS,"p"))
    p_element.set('class','ocr_par')
    wlen = len(wregions) - 1

    for i,wregion in enumerate(wregions):
        if last_line is None or not wregion.line in last_line:
            l_element.set('class','ocr_line')
            l_element.set('id','line_%d_%d' % (page_cnt,line_cnt))
            if last_line is not None:
                line_cnt += 1
                l_element.set('title',getBBox(l_element))
                p_element.append(l_element)
                l_element = ET.Element(ET.QName(HOCR_NS,"span"))
        w_element = ET.Element(ET.QName(HOCR_NS,"span"))
        w_element.set('class','ocrx_word')
        w_element.text = wregion.text
        w_element.set('title','bbox %s %s %s %s; x_wconf %d' %
                      (wregion.x0,wregion.y0,wregion.x1,wregion.y1,
                       wregion.conf))
        w_element.set('id','word_%d_%d' % (page_cnt,word_cnt))
        l_element.append(w_element)
        word_cnt += 1
        if last_line is not None: last_para = last_line.split(".")[0]
        last_line = wregion.line
        if i == wlen: last_line = '' # trigger if last word
        if i > 0 and not last_para in last_line:
            p_element.set('id','par_%d_%d' % (page_cnt,par_cnt))
            p_element.set('title',getBBox(p_element))
            block_element.append(p_element)
            block_element.set('id','block_%d_%d' % (page_cnt,block_cnt))
            block_element.set('title',getBBox(block_element))
            pg_element.append(block_element)
            block_cnt += 1
            p_element = ET.Element(ET.QName(HOCR_NS,"p"))
            p_element.set('class','ocr_par')
            par_cnt += 1
            block_element = ET.Element(ET.QName(HOCR_NS,"div"))
            block_element.set('class','ocr_carea')

    body_node.append(pg_element)
    parent_node.append(body_node)
    writeHocr(parent_node, HOCRfile)

parser = argparse.ArgumentParser()
arg_named = parser.add_argument_group("named arguments")
arg_named.add_argument("-b","--base", 
    help="base image")
arg_named.add_argument("-f","--folder", 
    help="input folder")
arg_named.add_argument("-c","--conf", default=50, type=int,
    help="set confidence number threshold for skipping regions")
arg_named.add_argument('-o', '--out', type=str,
    default="out.hocr",
    help="output hOCR file")
arg_named.add_argument("-t",'--text', action='store_true',
    default=False,
    help="process txt files instead of hocr")


args = parser.parse_args()

if args.base == None or not os.path.exists(args.base):
    print("missing base image, use '-h' parameter for syntax")
    sys.exit()

if args.folder == None or not os.path.exists(args.folder):
    print("missing folder, use '-h' parameter for syntax")
    sys.exit()

#get base image dimensions first
img = cv2.imread(args.base)
h,w = img.shape[:2]

print(f"Base image: {args.base} - {w}x{h}")
ocr_ext = ".hocr"
if args.text:
    ocr_ext = ".txt"

wregions = []
for para_file in sorted(glob.glob(args.folder + "/*" + ocr_ext)):
    #print("processing para:", para_file)
    if args.text:
        img = cv2.imread(para_file.replace('.jpg','').replace('.txt','.jpg'))
        th,tw = img.shape[:2]
        runThruText(para_file,tw/HNUM,th/HNUM,wregions)
    else:
        runThruHocr(para_file,args.conf,wregions)

sortOutWords(wregions,w,h,args.out,args.base)
