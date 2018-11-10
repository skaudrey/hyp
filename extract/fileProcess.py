#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 9:55
# @Author  : MiaFeng
# @Site    : 
# @File    : fileProcess.py
# @Software: PyCharm
__author__ = 'MiaFeng'
import pandas as pd
from Config import *
import re
from util.fileUtil import CIoUtil

class CRaw2CSV(object):
    def __init__(self):
        self.__ioUtil = CIoUtil()

    def loadChanIndex(self):
        fileName = 'chan_index.dat'
        chan_index = pd.read_table('%s/%s'%(getPath('data'),fileName),header=None)
        # print(chan_index.columns.values)
        cols = chan_index[0].values.tolist()
        # print(cols)
        cols = list(map(lambda x:'ch%d'%x,cols))

        return cols

    def getColumns(self,chanCols):
        return ['lon','lat','cloud','topo']+chanCols

    def fileStrip(self,filename,cols,filetype):
        '''
        strip lines with multiple blanks. save it as .csv file after striping and splitting.
        :param filename:
        :param cols:
        :param filetype:
        :return:
        '''
        filePath = '%s/%s%s'%(getPath(),filename,filetype)
        data = [[]]*len(cols) #init 2-dimensional list
        with open(filePath,'rb') as f:
            # print(f.readline())
            for line in f:
                line = str(line,encoding='utf-8').strip()
                line = re.sub(' +',' ',line)
                data_tmp = [[float(x)] for x in line.split(' ')]
                data = [*map(lambda x,y:x+y,data,data_tmp)]
                # for idx in np.arange(len(cols)):
                #     data[idx] = data[idx]+[data_tmp[idx]]
                #     # data[idx].append(data_tmp[idx])

        df = pd.DataFrame(data = dict(zip(cols,data)))
        self.__ioUtil.saveCSV(df,filename)
        return df
#
# def sortByLatAndLon(df,lonAscend,latDescend,saveFileName):
#     pass
#     # df = df.sort(by=)
#



if __name__=='__main__':
    columns = getColumns(loadChanIndex())
    # print(columns)
    # changeFileType(columns,'20150807_1','.txt','.csv')
    fileStrip('20150809_2',columns,'.txt')
