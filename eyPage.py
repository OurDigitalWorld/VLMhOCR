"""
eyPage.py - create paragraph images

Usage (see list of options):
    eyPage.py [-h]

This script extracta regions from an image, based
on a corresponding PAGE-XML file.

- art rhyno, u. of windsor & ourdigitalworld
"""

import xml.etree.ElementTree as ET
import argparse, glob, math, os, sys
import numpy as np
import cv2
import copy

MARGIN = 10 #num of pixels to add around outlines 

""" namespace for PAGE-XML """
PAGE_NS = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
ET.register_namespace('pc', PAGE_NS)

""" simple coordinate adjustment """
def expandNpArr(pts,extra,middle):
    # use the middle coordinates
    mid_x = math.ceil(middle[0])
    mid_y = math.ceil(middle[1])
    for pt in pts:
        # horizontal
        if pt[0] < mid_x:
            pt[0] -= extra
        elif pt[0] > mid_x:
            pt[0] += extra
        # vertical
        if pt[1] < mid_y:
            pt[1] -= extra
        elif pt[1] > mid_y:
            pt[1] += extra
        # clean up negative coordinates
        if pt[0] < 0: pt[0] = 0
        if pt[1] < 0: pt[1] = 0

""" adjust coordinates and define a bounding box """
def adjCoords(pts,extra):
    #find midpoint of blob
    mid = (pts.min(axis=0) + pts.max(axis=0)) * .5
    #now adjust
    expandNpArr(pts,extra,mid)
    #get the maximum points
    hvals = np.max(pts,axis=0)
    #get the minimun points
    lvals = np.min(pts,axis=0)
    #the goal is to put the blob in a box so sort out the needed numbers
    w = hvals[0] - lvals[0]

    h = hvals[1] - lvals[1]
    coords = {"x0":lvals[0],"y0":lvals[1],
              "x1":hvals[0],"y1":hvals[1]}

    return w, h, coords

""" extract ROI as a mask """
def extractMasked(img,pts,coords,clean_img):
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(mask, pts, 1)
    mask = mask > 0

    #this would make a black background:
    #    out = np.zeros_like(img)
    #but we will go with a white background
    out = np.ones(img.shape) * 255
    out[mask] = clean_img[mask]

    cropped_image = out[coords["y0"]:coords["y1"], coords["x0"]:coords["x1"]]
    cv2.rectangle(img,(coords["x0"],coords["y0"]), (coords["x1"],coords["y1"]),
        (0,255,0),3)

    return cropped_image

""" convert coordinates into ints """
def sortOutCoords(spts,pts):
    for spt in spts:
        pts.append([int(spt.split(',')[0]),
                    int(spt.split(',')[1])])

""" extract coordinates from specified element """
def getCoords(start,estr,isSep):

    pg_elems = []
    for elem in start.iterfind('.//{%s}%s' % (PAGE_NS,estr)):
        pts = []
        #paragraph is the 1st set of coordinates in PAGE-XML
        sortOutCoords(elem[0].attrib['points'].split(' '),pts)
        pg_elems.append(pts)

    return pg_elems

""" use coordinate block for region, roughly cooresponds to hOCR's paragraphs """
def sortOutPageTextRegions(PAGEfile,sourceFile,outFolder):
    tregions = [] # text regions

    tree = ET.ElementTree(file=PAGEfile)
    img = cv2.imread(sourceFile)
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    clean_img = img.copy()

    tregions = getCoords(tree,'TextRegion',False)
    print("extracting paragraph images.",end="",flush=True)
    for tregion in tregions:
        print(".", end="", flush=True)
        pts = np.asarray(tregion)
        #make room for margin
        w,h,coords = adjCoords(pts,MARGIN)
        cropped_image = extractMasked(img,pts,coords,clean_img)
        cropped_image = cv2.copyMakeBorder(
            cropped_image, top=MARGIN, bottom=MARGIN,
            left=MARGIN,right=MARGIN,borderType=cv2.BORDER_CONSTANT,
            value=[255,255,255])
        cropped_fn = "%s/para_%05d_%05d_%05d_%05d.jpg" % (outFolder,
            coords["x0"],
            coords["y0"],coords["x1"],coords["y1"])
        cv2.imwrite(cropped_fn,cropped_image)
    print("done!",flush=True)

parser = argparse.ArgumentParser()
arg_named = parser.add_argument_group("named arguments")
arg_named.add_argument("-f","--file", 
    help="input image, for example: imgs/my_image.tif")
arg_named.add_argument("-b","--border", default=10, type=int,
    help="adjust border value for extracted regions")
arg_named.add_argument('-o', '--output', type=str,
    default="imgs",
    help="output folder")

args = parser.parse_args()

if args.file == None or not os.path.exists(args.file):
    print("missing input image, use '-h' parameter for syntax")
    sys.exit()

#use filename to pull everything together
img_base = args.file.split(".")[0]
if not os.path.exists(img_base + ".xml"):
    print("No PAGR-XML file for image")
    sys.exit()

sortOutPageTextRegions(img_base + ".xml",args.file,args.output)
