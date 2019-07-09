# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:14:24 2019

@author: Saiyam_Jain
"""

def pred(df):
    ans = classifier.predict_classes(df)
    ans = np.argmax(to_categorical(ans), axis = 1)
    priority=np.argmax(to_categorical(ans), axis = 1)
    if priority==0:
        return(0, 'Priority-Low')
    elif priority==1:
        return(1, 'Priority-Medium')
    else:
        return(2, 'Priority-High')