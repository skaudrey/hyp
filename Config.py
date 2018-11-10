#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 9:55
# @Author  : MiaFeng
# @Site    : 
# @File    : Config.py
# @Software: PyCharm
__author__ = 'MiaFeng'

BASEPATH = '/home/skaudrey/code/python/hyp'
DATA_PREFIX = ['0807_1','0807_2','0808_1','0808_2','0809_1','0809_2']

PLOTLY_USR_NAME = 'Skaudrey'
PLOTLY_API_KEY = 'ooFX63kOai3l6spabIfA'

LAT_SIZE = 51
LON_SIZE = 51

def getPath(category = 'data'):
    subPath = ''
    if category in ['data','train','test','result']:
        subPath = category
    elif category == 'figData':
        subPath = 'data/df2fig'
    elif category=='model':
        subPath = 'result/model'
    elif category=='0807':
        subPath = 'result/0807'
    elif category == '0808':
        subPath = 'result/0808'
    elif category == '0809':
        subPath = 'result/0809'
    elif category == 'all':
        subPath = 'result/all'
    elif category == 'showfig':
        subPath = 'result/showfig'
    elif category in ['chan2','model2','model3','pca']:
        subPath = 'result/%s'%category
    elif category.startswith('chan'):
        subPath = 'result/%s'%category
    else:
        print("please check the category you set")
    return '%s/%s'%(BASEPATH,subPath)

