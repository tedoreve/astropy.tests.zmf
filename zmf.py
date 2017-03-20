# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:49:54 2016

@author: tedoreve
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mpath
from astropy import units as u
from astropy import wcs
import copy as cp

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

def plot_2Dfits(data,head,contrast):
    '''
    plot original continuum figure
    contrast is the contrast ratio, it would be better to use 1.
    Be careful!!! The nan values will be set to zeros.
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

def velocity(data,head):
    '''
    return spec velocity axis
    '''
    w   = wcs.WCS(head)
    pix = np.linspace(1,data.shape[0],data.shape[0])
    if head['NAXIS'] == 3:
        x,y,v   = w.wcs_pix2world(0,0,pix,0)
    if head['NAXIS'] == 4:
        x,y,v,s   = w.wcs_pix2world(0,0,pix,0,0)
    return v

def box(data,head,contrast,region,onoff,*args):
    '''
    plot box pixel coordinates
    '''
    l1,l2,b1,b2 = region  
    x1,y1,x2,y2 = coo_box(head,region)
    if data.ndim == 2:
        result0 = data[y1:y2,x1:x2]
        result1 = result0
    if data.ndim == 3:
        result0 = data[:,y1:y2,x1:x2] 
        result1 = data[args[0],y1:y2,x1:x2] 
    result1 = (result1-np.mean(result1))+np.mean(result1)*contrast
    
    plt.subplots() 
    plt.imshow(result1,origin='lower',interpolation='nearest',extent=[l2,l1,b1,b2])
    plt.colorbar()   
    
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (onoff[0], onoff[2])),
        (Path.CURVE4, (onoff[0], onoff[3])),
        (Path.CURVE4, (onoff[1], onoff[3])),
        (Path.CURVE4, (onoff[1], onoff[2])),
        (Path.CLOSEPOLY, (onoff[0], onoff[2])),   
        ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    x, y = zip(*path.vertices)
    plt.plot(x, y, 'go-')
    plt.xlim(l2,l1)
    plt.ylim(b1,b2)
    plt.grid()
    
    x1,y1,x2,y2 = coo_box(head,onoff)  
    if data.ndim == 2:
        result2 = data[y1:y2,x1:x2]
    if data.ndim == 3:
        result2 = data[:,y1:y2,x1:x2]
        
    return result2,result0


def circle(data,head,contrast,region,onoff,*args):
    '''
    plot circle pixel coordinates
    '''
    l1,l2,b1,b2 = region  
    x1,y1,x2,y2 = coo_box(head,region)
    if data.ndim == 2:
        result0 = data[y1:y2,x1:x2]
        result1 = result0
    if data.ndim == 3:
        result0 = data[:,y1:y2,x1:x2] 
        result1 = data[args[0],y1:y2,x1:x2] 
    result0 = (result0-np.mean(result0))+np.mean(result0)*contrast
    
    plt.subplots()
    plt.imshow(result1,origin='lower',interpolation='nearest',extent=[l2,l1,b1,b2])
    plt.colorbar()
    
    an = np.linspace(0, 2*np.pi, 100)
    plt.plot(onoff[2]*np.cos(an)+onoff[0], onoff[2]*np.sin(an)+onoff[1],'r')
    plt.xlim(l2,l1)
    plt.ylim(b1,b2)
    plt.grid()
    
    x,y,r =coo_circle(head,onoff)
    if data.ndim == 2:
        result2 = cp.copy(data[int(y-r):int(y+r),int(x-r):int(x+r)])
        for i in range(result2.shape[0]):
            for j in range(result2.shape[1]):
                if (i-result2.shape[0]/2)**2+(j-result2.shape[1]/2)**2 > r**2:
                    result2[i,j]=0
    if data.ndim == 3:
        result2 = cp.copy(data[:,int(y-r):int(y+r),int(x-r):int(x+r)])
        for i in range(result2.shape[1]):
            for j in range(result2.shape[2]):
                if (i-result2.shape[1]/2)**2+(j-result2.shape[2]/2)**2 > r**2:
                    result2[:,i,j]=0
                           
    return result2,result0

 

