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
    contrast is the contrast ratio, it would be better to use 1.
    name is the title of the image
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
    plt.imshow(np.log(result),origin='lower',interpolation='nearest',extent=[l1,l2,b1,b2])
    plt.grid()
    
def coo_box(head,region):
    '''
    get box pixel coordinates 
    l1,l2,b1,b2 = region
    '''
    l1,l2,b1,b2 = region
    w = wcs.WCS(head)
    if head['NAXIS'] == 2:
        x1,y1 = w.wcs_world2pix(l2,b1,0)
        x2,y2 = w.wcs_world2pix(l1,b2,0)   
    if head['NAXIS'] == 3:
        x1,y1,v = w.wcs_world2pix(l2,b1,0,0)
        x2,y2,v = w.wcs_world2pix(l1,b2,0,0)
    if head['NAXIS'] == 4:
        x1,y1,v,s = w.wcs_world2pix(l2,b1,0,0,0)
        x2,y2,v,s = w.wcs_world2pix(l1,b2,0,0,0)
    return x1,y1,x2,y2
        
def coo_circle(head,region):
    '''
    get circle pixel coordinates
    l,b,r = region
    please gurantee the head['CDELT1']=head['CDELT2']
    '''
    l,b,r = region
    r = r/np.abs(head['CDELT1'])
    w = wcs.WCS(head)
    if head['NAXIS'] == 2:
        x,y = w.wcs_world2pix(l,b,0)
    if head['NAXIS'] == 3:
        x,y,v = w.wcs_world2pix(l,b,0,0)
    if head['NAXIS'] == 4:
        x,y,v,s = w.wcs_world2pix(l,b,0,0,0)
    return x,y,r

 

