import sys
import csv
import os
import json
import bisect
import codecs
import pickle
import urllib.request
import shutil


# Filenames
f_requestedClasses = "requested_classes.csv"
f_labels = "OpenImagesV5/train-annotations-human-imagelabels.csv"
f_imageIds = "OpenImagesV5/train-images-with-labels-with-rotation.csv"
f_locatedImageIds = "located_image_ids.json"
f_locatedImageIdsHighConf = "located_file_ids_high_conf.json"
f_imagesOfInterest = "images_of_interest.pickle"
f_downloadLocation = "downloads"

# Run options
g_performFindFile = False # True = search the label file for ImageIDs with labels we want, False = skip search, assuming images_of_interst.pickle already exists
g_useHighConfIds = True   # True=only find images with confidence=1 in labels file, False=ignore confidence in labels file
g_findFileIds = False     # True=regenerate FileIds from master list, False=load FileIds from 'located' files
g_findImageInfos = False  # True=regenerate ImageInfos from master list, False=load ImageInfos from 'images_of_interest.pickle'
g_minFileSize = 5000000   # minimum file size to enter our list of images_of_interest

class ImageInfo(object):
    """Metadata about an image"""
    def __init__(self, id, addr, lbl, sz):
        self.ID = id;
        self.address = addr;
        self.label = lbl;
        self.size = sz;


def isIdInList(lst, x):
    'Locate the leftmost value in a exactly equal to x'
    i = bisect.bisect_left(lst, x)
    if i != len(lst) and lst[i] == x:
        return True
    return False

def findFileIds():
    if g_findFileIds:
        class_names = dict()
        class_ids = set()
        files = dict();
        files_high_confidence = dict();

        # read the names and classes we'd like to download
        with open(f_requestedClasses, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names[row['class_id']] = row['name'];
                class_ids.add(row['class_id'])
                files[row['name']] = []
                files_high_confidence[row['name']] = []

        # find the ImageIDs that have the class_id we want
        cnt = 0
        with open(f_labels, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cnt = cnt + 1
                if row['LabelName'] in class_ids:
                    class_name = class_names[row['LabelName']]
                    files[class_name].append(row['ImageID'])
                    if int(row['Confidence']) > 0:
                        files_high_confidence[class_name].append(row['ImageID'])
                if (cnt % 10000) == 0:
                    print("processed " + str(cnt) + " records so far...")

        # write out the files and files_high_confidence dicts to json, so we don't
        # do the above search again
        with open(f_locatedImageIds, 'w') as fp:
            json.dump(files, fp, indent=4)
        with open(f_locatedImageIdsHighConf, 'w') as fp:
            json.dump(files_high_confidence, fp, indent=4)
    else:
        # read the files and files_high_confidence dicts to json, so we don't
        # do the above search again
        if g_useHighConfIds:
            with open(f_locatedImageIdsHighConf) as fp:
                files_high_confidence = json.load(fp)
        else:
            with open(f_locatedImageIds) as fp:
                files = json.load(fp)

    if g_useHighConfIds:
        return files_high_confidence
    else:
        return files

def findImageInfos(ids):
    if g_findImageInfos:
        imgs = dict();
        for label in ids:
            imgs[label] = [];
        # read the master list
        cnt = 0
        with open(f_imageIds, encoding='UTF8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                idCur = row['ImageID']
                # check every label that we're looking for to see if this image is part of that label
                for label in ids:
                    if isIdInList(ids[label], idCur) and int(row['OriginalSize']) > g_minFileSize:
                        imgs[label].append(ImageInfo(row['ImageID'], row['OriginalURL'], label, int(row['OriginalSize'])))
                cnt = cnt + 1
                if (cnt % 10000) == 0:
                    print("searched " + str(cnt) + " lines of images so far...")
    
        for label in imgs: # sort by biggest files first
            imgs[label].sort(key=lambda img: img.size, reverse=True)

        # use pickle because json doesn't (easily) store the ImageInfo class instances...
        with open(f_imagesOfInterest, 'wb') as fp:
            pickle.dump(imgs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f_imagesOfInterest, 'rb') as fp:
            imgs = pickle.load(fp)
    return imgs

def downloadImages(imgs):
    for label in imgs:
        dir = os.path.join(f_downloadLocation, label)
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    counts = dict()
    for label in imgs:
        counts[label] = 0

    still_working = True;
    while still_working:
        still_working = False
        for label in imgs:
            idx = counts[label]
            if idx < len(imgs[label]):
                info = imgs[label][idx]
                still_working = True
                
                fname = os.path.join(f_downloadLocation, label, info.ID + '.jpg')
                if os.path.exists(fname):
                    print(f"{label:10} ({idx}): Already downloaded to {fname}") 
                else:
                    try:
                        tmp_fname = os.path.join(f_downloadLocation, label, '_pending.jpg')
                        print(f"{label:10} ({idx}): downloading {info.address} to {fname}...") 
                        urllib.request.urlretrieve(info.address, tmp_fname)
                        os.rename(tmp_fname, fname)
                    except:
                        print(f"{label:10} ({idx}):   unable to download {info.address} to {fname}...") 
                counts[label] = counts[label] + 1

def main():
    """main function"""
    global g_performFindFile

    if g_findImageInfos:
        g_performFindFile = True

    if g_performFindFile:
        ids = findFileIds()
    else:
        ids = []

    imgs = findImageInfos(ids)

    downloadImages(imgs)

if __name__ == '__main__':
    main()
