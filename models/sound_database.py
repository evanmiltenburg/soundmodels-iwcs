# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 12:38:04 2014

@author: alessandro

The present module contains functions to prepare a sound db for the BoAW's and
related experiments.
It contains functions for:
    -sound conversion to .wav
    -create a .txt with the stats of the DB and a description of the single 
    sounds (besides the one already provided by the source of the DB, eg: FreeSound.org)
    -create a SoundDB_Complet and a TrainingDB folder
"""

import glob2
import numpy as np
import os
from pydub import AudioSegment
import glob
import shutil
from scipy import signal as sg
from scipy.io import wavfile
import random
import unicodecsv
import string
from collections import Counter


###############################################################################
# Database content and preprocessing tools
###############################################################################


def soundlist(maindir): 
    #From a directory containing the sounds dataset, it returns a numpy array listing all the sounds.
    soundlist = np.sort(glob.glob(maindir + "/*.*"))
    return np.array(soundlist)

def soundnamelist(soundlist):
    #From a numpy array listing sounds (output of soundlist()), it returns a numpy array listing all the sound names (i.e. id's).
    soundnamelist = [os.path.splitext(os.path.basename(line))[0] for line in soundlist]
    return np.array(soundnamelist)
  
def resampling(soundlist, samplerate):
    #Take a list of sounds to be resampled to samplerate.
    for s in soundlist:
        rate, data = wavfile.read(s)
        print rate
        if rate is not samplerate:
            resampled = sg.resample(data, samplerate)
            wavfile.write(s,samplerate,resampled)
            
def converttowav(soundlist):
    #Take a list of sounds and make sure they are in .wav format (eventually convert them).
    for s in soundlist:
        filename, extension = os.path.splitext(s)
        if extension != '.wav':
            print extension[1:] 
            sound = AudioSegment.from_file(s, extension[1:])
            sound.export(filename+".wav", format="wav")
            os.remove(s)




###############################################################################
# Training list generator
###############################################################################


def trainiglist(soundlist, nr):
    #From the list of sounds in the database, create a random selection of nr sounds for training.
    nr_sounds = soundlist.shape[0]
    random_index_list = random.sample(xrange(nr_sounds), nr)
    traininglist = np.array([soundlist[index] for index in random_index_list])
    return traininglist




###############################################################################
# Database descriptors
###############################################################################


def taglist(tagfile):
    #return a list of tags from the tag file
    with open(tagfile, 'r') as f:
        tags = set(list([word for line in f for word in line.split('\t')[1].strip('\n').split(' ') if not word.isdigit()]))
    return sorted(tags)   


def soundid_words(tagfile): 
    #return a dictionary sound_id : tags
    with open(tagfile, 'r') as f:
        lines = f.readlines()
        id_list = [line.split('\t')[0] for line in lines]
        tagging_list = [line.split('\t')[1].strip('\n').split(' ') for line in lines]  
    return {id_list[i]: tagging_list[i] for i in range(len(id_list))}


def word_soundid(wordlist, soundidwords): 
    #return a dictionary tag : sound_id's
    return {wordlist[i]: [sound for sound in soundidwords if wordlist[i] in soundidwords[sound]] for i in range(len(wordlist))}
