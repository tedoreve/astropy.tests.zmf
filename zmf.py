# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:49:54 2016

@author: tedoreve
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import wcs

def distance2diameter(distance,angle):
    '''distance(kpc) angle(arcmin) return(pc)'''
    return distance*1000*angle/60/180*np.pi
    
def conversion(v,bmaj,bmin):
    '''
    v(GHz),bmaj(deg),bmin(deg),retrun(Jy/beam -> K)
    '''
    bmaj = bmaj*u.deg
    bmin = bmin*u.deg
    fwhm_to_sigma = 1./(8*np.log(2))**0.5
    beam_area = 2.*np.pi*(bmaj*bmin*fwhm_to_sigma**2)
    freq = v*u.GHz
    equiv = u.brightness_temperature(beam_area, freq)
    return u.Jy.to(u.K, equivalencies=equiv)  

def plot_fits(data,head,contrast,name):
    '''
    plot original continuum figure
    '''
    w = wcs.WCS(head)
    if head['NAXIS'] == 2:
        l1,b1 = w.wcs_pix2world(0,0,0)
        l2,b2 = w.wcs_pix2world(data.shape[1],data.shape[0],0)
    if head['NAXIS'] == 3:
        l1,b1,v = w.wcs_pix2world(0,0,0,0)
        l2,b2,v = w.wcs_pix2world(data.shape[1],data.shape[0],0,0)    
    if head['NAXIS'] == 4:
        l1,b1,v,s = w.wcs_pix2world(0,0,0,0,0)
        l2,b2,v,s = w.wcs_pix2world(data.shape[1],data.shape[0],0,0,0)

    
    result = np.nan_to_num(data)
    result = (result-np.mean(result))+np.mean(result)*contrast

    plt.subplots() 
    plt.title(name)
    plt.imshow(np.log(result),origin='lower',interpolation='nearest',extent=[l2,l1,b1,b2])
    plt.grid()

