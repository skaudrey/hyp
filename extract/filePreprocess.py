# -*- coding: utf-8 -*-
# @File     : filePreprocess.py
# @Software : PyCharm   
__time__ = '9/20/184:58 PM'
__author = 'MiaFeng'

import pandas as pd
import re
from Config import *
import numpy as np


def getColumnsNames():
    fileName = 'chain_index.dat'
    chan_index = pd.read_table("%s/%s"%(getPath(),fileName),header=None)
    cols = chan_index[0].values.tolist()
    cols = list(map(lambda x:'ch%d'%x,cols))
    return ['lat','lon','cloud','topo']+cols

def fileStrip(filename,cols,filetype):
    filePath = '%s/%s%s'%(getPath(),filename,filetype)
    data = [[]]*len(cols) # init 2D-list
    with open(filePath,'rb') as f:
        for line in f:
            line = str(line,encoding='utf-8').strip()
            line = re.sub(' +',' ',line) # replace blanksapces(either one or many) by one blankspace
            data_tmp = [[float(x)] for x in line.split(' ')]
            data = [*map(lambda x,y:x+y,data,data_tmp)]
    df = pd.DataFrame(data=dict(zip(cols,data)))
    saveCSV(df,filename)
    return df


if __name__=='__main__':
    cols = getColumnsNames()

    fileStrip('2018')