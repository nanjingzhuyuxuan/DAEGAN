# -*- coding: utf-8 -*-

import random
import re
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import preprocessing

import data_fill
import cfgan

import warnings
import math
warnings.filterwarnings("ignore")


def select_negative_items(realData, num_pm,wake_negativeBatch):

    data = np.array(realData)
    n_items_pm = np.zeros_like(data)
    for i in range(data.shape[0]):
        all_item_index = wake_negativeBatch[i]
        random.shuffle(all_item_index)
        n_item_index_pm = all_item_index[0: num_pm]
        n_items_pm[i][n_item_index_pm] = 1
    return n_items_pm

def computeTopN(groundTruth, result, topN):
    # 准确率
    result = result.tolist()
    for i in range(len(result)):
        result[i] = (result[i], i)
    result.sort(key=lambda x: x[0], reverse=True)
    hit = 0
    for i in range(topN):
        if (result[i][1] in groundTruth):
            hit = hit + 1

    p = hit / topN

    r = hit / len(groundTruth)

    DCG = 0
    IDCG = 0

    for i in range(topN):
        if (result[i][1] in groundTruth):
            DCG += 1.0 / math.log(i + 2)

    for i in range(topN):
        IDCG += 1.0 / math.log(i + 2)
    ndcg = DCG / IDCG
    return p, r, ndcg


def main(userCount, itemCount, testSet, trainVector, trainMaskVector, topN, epochCount, alpha, wake_negativeList,
         trainVector_eu):
    result_precision = np.zeros((1, 2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = cfgan.generator(itemCount).to(device)
    D = cfgan.discriminator(itemCount).to(device)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
    G_step = 5
    D_step = 2
    batchSize_G = 32
    batchSize_D = 32

    for epoch in range(epochCount):

        for step in range(G_step):

            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))

            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G])).to(device)
            eu_f = trainVector_eu[leftIndex:leftIndex + batchSize_G]
            wake_negativeBatch = wake_negativeList[leftIndex:leftIndex + batchSize_G]

            n_items_pm = select_negative_items(realData, pro_PM,wake_negativeBatch)
            ku_zp = Variable(torch.tensor(n_items_pm )).to(device)
            eu_f = Variable(torch.tensor(eu_f)).to(device)
            realData_zp = Variable(torch.ones_like(realData)).to(device) * eu+ Variable(torch.zeros_like(realData)).to(device) * eu_f+Variable(torch.full_like(realData,0.1)).to(device) *ku_zp

            fakeData = G(realData.to(device)).to(device)
            fakeData_ZP = fakeData * (eu + eu_f +ku_zp)
            fakeData_result = D(fakeData_ZP)

            g_loss = np.mean(np.log(1. - fakeData_result.detach().cpu().numpy() + 10e-5)) + alpha * regularization(
                fakeData_ZP, realData_zp)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        for step in range(D_step):

            leftIndex = random.randint(1, userCount - batchSize_D - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_D]))
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G])).to(device)
            eu_f = trainVector_eu[leftIndex:leftIndex + batchSize_G]

            n_items_pm = select_negative_items(realData, pro_PM,wake_negativeBatch)
            ku = Variable(torch.tensor(n_items_pm)).to(device)

            fakeData = G(realData.to(device)).to(device)
            eu_f = Variable(copy.deepcopy(eu_f)).to(device)
            fakeData_ZP = fakeData * (eu  + ku +eu_f)

            fakeData_result = D(fakeData_ZP)
            realData_result = D(realData.to(device))
            d_loss = -np.mean(np.log(realData_result.detach().cpu().numpy() + 10e-5) + np.log(
                1. - fakeData_result.detach().cpu().numpy() + 10e-5)) + 0 * regularization(fakeData_ZP, realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        if (epoch % 1 == 0):
            n_user = len(testSet)
            precisions_5 = 0
            precisions_10 = 0
            ndcg_5 = 0
            ndcg_10 = 0
            index = 0

            for testUser in testSet.keys():
                data = Variable(copy.deepcopy(trainVector[testUser])).to(device)
                result = G(data.reshape(1, itemCount)) + Variable(copy.deepcopy(trainMaskVector[index])).to(device)
                result = result.reshape(itemCount)

                precision, recall, ndcg = computeTopN(testSet[testUser], result, 5)
                precisions_5 += precision
                ndcg_5 += ndcg

                precision, recall, ndcg = computeTopN(testSet[testUser], result, 10)
                precisions_10 += precision
                ndcg_10 += ndcg
                index += 1

            precisions_5 = precisions_5 / n_user
            precisions_10 = precisions_10 / n_user
            ndcg_5 = ndcg_5 / n_user
            ndcg_10 = ndcg_10 / n_user
            result_precision = np.concatenate((result_precision, np.array([[epoch, precisions_5]])), axis=0)
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{},precision:{},ndcg5:{},ndcg10:{}'.format(epoch,
                                                                                                                 epochCount,
                                                                                                                 d_loss.item(),
                                                                                                                 g_loss.item(),
                                                                                                                 precisions_5,
                                                                                                                 precisions_10,
                                                                                                                 ndcg_5,
                                                                                                                 ndcg_10))

    return result_precision


import time
def result_plt(result_precision,name):
    plt.figure()
    plt.title("the precision of CDAE-WGAN")
    plt.xlabel('epoch')
    plt.plot(result_precision[:, 0], result_precision[:, 1], "r-*", label='precision')
    plt.ylim([0, 0.6])
    plt.legend()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(name+timestr + '.png')
    plt.show()
if __name__ == '__main__':
    topN = 5
    epochs = 1000
    pro_PM = 100
    alpha = 0.1

    config = {
        '100k': {
            'train': '../../code_ITVeALS/data/movielens 100k/train.csv',
            'fill': '../../code_ITVeALS/data/movielens 100k/fill8.csv',
            'test': '../../code_ITVeALS/data/movielens 100k/test.csv',
        },
        'latest': {
            'train': '../../code_ITVeALS/data/movielens latest/train.csv',
            'fill': '../../code_ITVeALS/data/movielens latest/fill8.csv',
            'test': '../../code_ITVeALS/data/movielens latest/test.csv',
        },
        'am': {
            'train': '../../code_ITVeALS/data/amazon CDs/train.csv',
            'fill': '../../code_ITVeALS/data/amazon CDs/fill8.csv',
            'test': '../../code_ITVeALS/data/amazon CDs/test.csv',
        },
        'cd_movies': {
            'train': '../../code_ITVeALS/data/cd_movies/train.csv',
            'fill': '../../code_ITVeALS/data/cd_movies/fill6.csv',
            'test': '../../code_ITVeALS/data/cd_movies/test.csv',
        },
        'prime': {
            'train': '../../code_ITVeALS/data/prime/train.csv',
            'fill': '../../code_ITVeALS/data/prime/fill8.csv',
            'test': '../../code_ITVeALS/data/prime/test.csv',
        },
        'ml1m': {
            'train': '../../code_ITVeALS/data/ml1m/train2.csv',
            'fill': '../../code_ITVeALS/data/ml1m/fill4_2.csv',
            'test': '../../code_ITVeALS/data/ml1m/test2.csv',
        },
    }

    data_name = '100k'
    data_config = config[data_name]
    trainSet, train_use, train_item = data_fill.loadTrainingData(data_config['train'], "\t")
    testSet, test_use, test_item = data_fill.loadTestData(data_config['test'], "\t")
    fillSet, fill_use, fill_item = data_fill.loadFillData(data_config['fill'], "\t")

    userCount = max(train_use, test_use)
    itemCount = max(train_item, test_item)
    userList_test = list(testSet.keys())
    trainVector, trainMaskVector, batchCount, wake_negativeList, trainVector_eu = data_fill.to_Vectors(trainSet, fillSet,
                                                                                                    userCount,
                                                                                                    itemCount,
                                                                                                    userList_test,
                                                                                                    "userBased")

    result_precision, result_last = main(userCount, itemCount, testSet, trainVector, trainMaskVector, topN, epochs,
                                         alpha, wake_negativeList, trainVector_eu)
    result_precision = result_precision[1:, ]
    result_plt(result_precision,data_name)

