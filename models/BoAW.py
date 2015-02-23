# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:21:56 2014

@author: alessandro lopopolo

This module implement the Bag of Audio Words model following the pipeline described in [Pancoast 20xx]. It takes a set of sounds, creates a audio vocabulary over a training subset of these sounds and encode the rest of the sounds as distributional vector over the audio words created in the training step. 

The module is composed by two types of functions: vocabulary_*, and encoding_*.

	vocabulary_* functions create a vocabulary, i.e. a list of Audio Words, out of a set of sounds (training set) using Mel-frequency 		Cepstral Coefficients (MFCC's) and a clustering algorithm.

	[At the present time (15/09/2014) only k-mean clustering is implemented (function vocabulary_kmean())]

	encoding_* functions take the vocabulary (BoAW) created by the vocabulary_* function and encode a set of sounds (possibly non 		overlapping with the training set) as a distribution over the Auditory Words.

	[At the present time (15/09/2014) only euclidean encoding is implemented (function encoding_euclidean())]

TO DO:	-Implement GMM vocabulary building function (vocabulary_GMM()),
	-Implement Fisher encoding function (encoding_Fisher()),
	-Consider other auditory features in the training step (vocabulary_*).
"""

from features import mfcc
from scipy.io import wavfile
from scipy.cluster.vq import kmeans, kmeans
from sklearn.cluster import k_means
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import glob
import os.path
import re


def vocabulary_kmean(traininglist, numberofwords, winlen=0.025, winstep=0.01, numcep=13):
    """
    Create a vocabulary of Audio Words (BoAW) from a training set of sounds using MFCC features and a k-means clustering algorithm.

    Args:
      traininglist (numpy array): The list of sounds to be used to create the BoAW.
      numberofwords (int): The desired number of AW's to be in the BoAW.
      winlen (float, optional): The width in seconds of the sampling window for the extraction of the MFCC features. Default to 0.025s.
      winstep (float, optional): The step between successive windows in seconds. Default to 0.01s.
      numcep (int, optional): The number of MFCC extracted in each window (i.e. the size of the Audio Words). Default to 13.

    Return:
      auditorywords (numpy array [numberofwords x numcep]): the BoAW vocabulary.
    """
    
    print 'VOCABULARY BUILDING STARTS'
    
    awnames = np.arange(1,(numberofwords+1)).T
    
    #The soon to be filled set of all the features extracted from the training set
    features = np.empty((0,13)) 
    
    #Feature extraction
    for s in traininglist:
        print s
        rate, data = wavfile.read(s)
        featurization = mfcc(data, samplerate=rate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=26, nfft=512, 
                             lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
        features = np.concatenate([features,featurization])
    print s
    print "\n Feature extraction done\n\n"
    
    
    #clustering (number of clusters = intended number of AW's)
    auditorywords,labels,inertia = k_means(features, numberofwords, init='k-means++', precompute_distances=True, n_init=3, max_iter=1000, verbose=True, tol=0.0001, random_state=None, copy_x=True, n_jobs=1)
    
    return auditorywords, awnames
    


    

def encoding_euclidean(soundlist, auditorywords, numberofwords, winlen=0.025, winstep=0.01, numcep=13):
    """
    Transform the sounds in input in vectors of distributions over a AW vocabulary. The algorith takes each sound 
    and extract a MFCC feature for each window, it assign each of these feature to a AW in the vocabulary and returns a vector counting 
    how many time a AW occurs in the sound at hand.

    Args:
      sounddir (str): The location of the sounds to be encoded using the BoAW.
      auditorywords (numpy array): The AW vocabulary (BoAW).
      numberofwords (int): The number of AW in the BoAW.
      winlen (float, optional): The width in seconds of the sampling window for the extraction of the MFCC features. Default to 0.025s.
      winstep (float, optional): The step between successive windows in seconds. Default to 0.01s.
      numcep (int, optional): The number of MFCC extracted in each window (i.e. the size of the Audio Words). Default to 13.

    Return:
      encodedsounds (numpy array [sounds x auditorywords]): the encoded sounds.
    """
    
    print 'SOUND ENCODING STARTS' 
    
    #The soon to be filled set of encoded sounds
    encodedsounds = np.empty((0, numberofwords), dtype=int)    
    
    #Feature extraction and assigment to AW's
    for s in soundlist:
        print s
        rate, data = wavfile.read(s)
        featurization = mfcc(data, samplerate=rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
        sound_encoding = np.zeros((1,numberofwords))
        distances = euclidean_distances(featurization, auditorywords, Y_norm_squared=None, squared=False)
        
        for column in distances:
            index = np.where(column == min(column))[0][0]
            sound_encoding[0,index] = sound_encoding[0,index]+1
        
        encodedsounds = np.concatenate([encodedsounds, sound_encoding], axis=0)
        
    return encodedsounds
