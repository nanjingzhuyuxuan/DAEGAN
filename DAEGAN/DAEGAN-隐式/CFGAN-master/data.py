# -*- coding: utf-8 -*-

from collections import defaultdict
import torch
import pandas as pd

def loadTrainingData(trainFile,splitMark):
    trainSet = defaultdict(dict)
    max_u_id = -1
    max_i_id = -1
    for line in open(trainFile):
        userId, itemId,rating, _ = line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        trainSet[userId].update({itemId:rating})
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Training data loading done" )
    return trainSet,userCount,itemCount


def loadTestData(testFile,splitMark):
    testSet = defaultdict(dict)

    max_u_id = -1
    max_i_id = -1
    for line in open(testFile):

        userId, itemId, rating ,_= line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        testSet[userId].update({itemId:rating})
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Test data loading done")
    return testSet,userCount,itemCount

def loadFillData(fillFile, splitMark):
    fillSet = defaultdict(dict)
    max_u_id = -1
    max_i_id = -1
    for line in open(fillFile):
        userId, itemId, rating  = line.strip().split('\t')
        userId = int(userId)
        itemId = int(itemId)
        fillSet[userId].update({itemId:rating})
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id+1
    itemCount = max_i_id+1
    print("Fill data loading done")
    return fillSet, userCount, itemCount

def to_Vectors(trainSet,userCount, itemCount, userList_test):

    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount
    trainDict = defaultdict(lambda: [0] * itemCount)

    for userId, i_list in trainSet.items():
        for itemId,rate in i_list.items():
            testMaskDict[userId][itemId] = -9999
            trainDict[userId][itemId] = 1
    trainVector = []

    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])
    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])

    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount

