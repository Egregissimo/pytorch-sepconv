import numpy as np
from numpy import linalg as la
import cv2 as cv
import math as m
import random as rnd
import time
import os
from glob import glob

# I need to tell if my flow is too little or too high
# to do so I compare the histograms of successive frames
def checkFlow(triplet, min_threshold, max_threshold):
    hist0 = cv.calcHist([triplet[0]],[0],None,[256],[0,256])
    hist1 = cv.calcHist([triplet[1]],[0],None,[256],[0,256])
    hist2 = cv.calcHist([triplet[2]],[0],None,[256],[0,256])
    motion0 = la.norm(hist0 - hist1)
    motion1 = la.norm(hist1 - hist2)
    return motion0 < max_threshold and motion1 < max_threshold and motion0 > min_threshold and motion1 > min_threshold

# interface for defining a stream of opecv images to be elaborated
class framesStream:
    def isOpened(self):pass
    def read(self):pass
    def release(self):pass

class videoStream(framesStream):
    stream = None # the stream is a opencv video stream

    def __init__(self, stream):
        self.stream = stream
    def isOpened(self):
        return self.stream.isOpened()
    def read(self):
        return self.stream.read()
    def release(self):
        return self.stream.release()

class imagesStream(framesStream):
    file_names = [] # the stream is a list of images to be opened
    
    def __init__(self, file_names):
        self.file_names = file_names
    def isOpened(self):
        return len(self.file_names) > 0
    def read(self):
        return (self.isOpened(), cv.imread(self.file_names.pop()))
    def release(self):
        return True

# Read stream, group frames in groups of 3, crop randomly to get patches of cropsize x cropsize pixels.
# Take only some patches and discard patches that have too little or too high motion.
def extractFromStream(stream, file_name, output_folder, frame_spacing, crops_per_frame, fps, width, height):
    crop_size = 150
    time_elapsed = 0
    time_delta = 0
    frame_number = 0
    # list containing the crops to be done to the current triple of frames
    random_crops = [] 
    # list containing corresponding patches for 3 successive frames
    frames_triplets = ([], [], [])

    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            print("stream end?")
            break

        #take a triplet every second
        if time_delta > frame_spacing:
            #time.sleep(.1)
            if frame_number < 3:
                # randomly decide crops for the triple, crops_per_frame times
                if frame_number == 0:
                    for _ in range(crops_per_frame):
                        random_crops.append((rnd.randint( 0, height - crop_size), rnd.randint(0, width - crop_size)))
                # crop the current frame as decided
                for i in range(crops_per_frame):
                    crop_img = frame[random_crops[i][0]:random_crops[i][0] + crop_size, random_crops[i][1]:random_crops[i][1]+crop_size]
                    frames_triplets[frame_number].append(crop_img)

                frame_number += 1
            else:   # already got 3 frames, I need to reset the timer and save frames
                #cv.imshow('frame', frame)
                frame_number = 0
                time_delta = 0

                # check all extracted patches and save those with right flow
                for i in range(crops_per_frame):
                    frames_triplet = (frames_triplets[0][i], frames_triplets[1][i], frames_triplets[2][i])
                    isFlowGood = checkFlow(frames_triplet, 250, 4000)
                    if isFlowGood:
                        #cv.imshow('crop0', frames_triplet[0])
                        #cv.imshow('crop1', frames_triplet[1])
                        #cv.imshow('crop2', frames_triplet[2])
                        cv.imwrite(output_folder + file_name + "_" +
                                    str(time_elapsed) + "_" + str(i) + "_" + '0.jpg', frames_triplet[0])
                        cv.imwrite(output_folder + file_name + "_" +
                                    str(time_elapsed) + "_" + str(i) + "_" + '1.jpg', frames_triplet[1])
                        cv.imwrite(output_folder + file_name + "_" +
                                    str(time_elapsed) + "_" + str(i) + "_" + '2.jpg', frames_triplet[2])

                frames_triplets = ([], [], [])

        time_elapsed += 1/fps
        time_delta += 1/fps
        if cv.waitKey(1) == ord('q'): break
    stream.release()

def extractFromVideo(file_name, output_folder, frame_spacing, crops_per_frame):
    stream = cv.VideoCapture(file_name)
    fps = stream.get(cv.CAP_PROP_FPS)
    width  = stream.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = stream.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`

    extractFromStream(videoStream(stream), file_name.replace('\\',''), output_folder, frame_spacing, crops_per_frame, fps, width, height)

def extractFromImages(folder_name, output_folder, frame_spacing, crops_per_frame):
    file_names = [folder_name + file for file in os.listdir(folder_name)]
    file_names.reverse()
    fps = 24
    firstImg = cv.imread(file_names[0])
    height, width = firstImg.shape[:2]
    extractFromStream(imagesStream(file_names), folder_name.split("\\")[-2], output_folder, frame_spacing, crops_per_frame, fps, width, height)

#extractFromVideo("drift.mp4", ".\\frames\\", 0.5, 15)

for root, dirs, files in os.walk(".\\UCF-101"):
    for file in files:
        if file.endswith(".avi"):
            print("processing: " + file)
            extractFromVideo(os.path.join(root, file), ".\\frames\\", 0.3, 5)

#imageFolders = glob(".\\input_images\\annotations\\*\\")
#for imagesFolder in imageFolders:
#    extractFromImages(imagesFolder, ".\\frames\\", .1, 10)