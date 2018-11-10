#!/usr/bin/env python
# encoding: utf-8
'''
@author: MiaFeng
@contact: skaudrey@163.com
@file: fileUtil.py
@time: 2018/9/26 14:34
@desc:
'''
from Config import *
import joblib
import pandas as pd
import numpy as np

def makeFileDescrip(modelName, learning_rate, epoch, batch_size):
    return 'weights_%s_lr%.2f_epc%d_batch%d' % (
        modelName,learning_rate, epoch, batch_size)

class CIoUtil(object):
    '''
    Handle the input and output things in many file types.
    '''
    def __init__(self):
        pass

    def saveCSV(self,df, filename='', category='data'):
        filename = '%s/%s.csv' % (getPath(category), filename)
        try:
            print("The dataframe you want to save has <%d> lines" % len(df))
            df.to_csv(filename, sep=',', encoding='utf-8', index=None)
        except:
            print('check filename or dataframe')
        print('save file --> %s' % filename)

    def saveArray(self,np_arr, fileName):
        fileName = "%s/%s_shape_%s.bin" % (getPath(), fileName, np_arr.shape)
        np_arr.tofile(fileName)
        print('the shape of saved array --> ')
        print(np_arr.shape)
        print('save file --> %s' % fileName)

    def loadArray(self,fileName):
        data_shape = [int(itm) for itm in fileName[fileName.find('(') + 1:fileName.rfind(')')].split(',')]
        data = np.fromfile(fileName,dtype=np.float16)
        data.shape = (data_shape[0], data_shape[1], data_shape[2],data_shape[3])

        return data

    def saveModel(self,model, saveFileName):
        saveFileName = "%s/%s.m" % (getPath('model'), saveFileName)
        joblib.dump(model, saveFileName)
        print('Done for saving file as --> %s' % saveFileName)

    def loadModel(self,loadFileName):
        loadFileName = "%s/%s.m" % (getPath('model'), loadFileName)
        data = joblib.load(loadFileName)
        print('Done for load file --> %s' % loadFileName)
        return data

    def changeFileType(self,chanCols, filename, rawFileType, outFileType):
        df = pd.read_csv('%s/%s%s' % (getPath(), filename, rawFileType), delimiter=' ', header=None)
        df[0] = df[0].strip()
        # print(df.head())
