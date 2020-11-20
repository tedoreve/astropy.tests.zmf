# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:49:54 2016

@author: tedoreve
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from astropy import units as u
from astropy import wcs
import astropy.constants as const
import astropy.units as un
from astropy.io import fits
import copy as cp
import time
from functools import wraps

def T2I(temperature, v, bmaj, bmin):
    '''
    calculate intensity
    temperature(K) v(GHz) bmaj(arcsec) bmin(arcsec)
    return 1 (Jy/beam)
    '''
    fwhm_to_sigma = 1./(8*np.log(2))**0.5
    beam_area = 2.*np.pi*(bmaj*un.arcsec*bmin*un.arcsec*fwhm_to_sigma**2)
    freq = v*u.GHz
    equiv = u.brightness_temperature(freq)

    return temperature/(u.Jy/beam_area).to(u.K, equivalencies=equiv)


def mdotrhov(parameter, r=0, mdot=0, rho=0, v=0):
    '''
    calculate mdot, rho, v in stellar winds
    parameter(choose in mdot, rho, v)
    r(pc) m_dot(M_sun/yr) rho('cm^-3') v(km/s)
    return based on the parameter chosen
    '''
    if parameter == 'mdot':
        return (4*np.pi*(r*un.pc)**2*rho*un.cm**-3*const.m_p*v*un.km/un.s).to('M_sun/yr')
    if parameter == 'rho':
        return (mdot*un.M_sun/un.yr/(4*np.pi*(r*un.pc)**2*const.m_p*v*un.km/un.s)).to('cm^-3')
    if parameter == 'v':
        return (mdot*un.m_sun/un.yr/(4*np.pi*(r*un.pc)**2*rho*un.cm**-3*con.m_p)).to('km/s')

def T2S(temperature, gamma, m = 1):
    '''
    calculate sonic speed in ideal gas,
    temperature(K),
    gamma(adiabatic index, 5/3 for 1 atom, 7/5 for 2 atoms, 4/3 for 3 atoms),
    m(the mass of a single molecule/1 H atom mass, default 1),
    return (km/s)
    '''
    return np.sqrt(temperature*un.K*gamma*const.k_B/m/const.m_p).to('km/s')

def S2T(sonicspeed, gamma, m = 1):
    '''
    calculate temperature in ideal gas,
    sonicspeed(km/s)
    gamma(adiabatic index, 5/3 for 1 atom, 7/5 for 2 atoms, 4/3 for 3 atoms)
    m(the mass of a single molecule/1 H atom mass, default 1)
    return (K)
    '''
    return ((sonicspeed*un.km/un.s)**2*m*const.m_p/gamma/const.k_B).to('K')

def cf(temperature,plot=False):
    '''
    calculate cooling function lambda(T)
    temperature(K), return(erg cm^-3 s^-1)
    array input is accessible
    '''
    cooling = np.loadtxt('cooling.dat')
    if plot:
        plt.loglog(cooling[:,0],cooling[:,1])
        plt.loglog(temperature,np.interp(temperature, cooling[:,0],cooling[:,1]),'o')

    return np.interp(temperature, cooling[:,0],cooling[:,1])

def tt(function):
    '''
    time test decorator
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.__name__, str(t1-t0))
               )
        return result
    return function_timer



def distance2diameter(distance, angle):
    '''
    distance(kpc) angle(arcmin) return(pc)
    '''
    return distance*1000*angle/60/180*np.pi

def diameter2arcmin(distance, diameter):
    '''
    distance(kpc) diameter(pc) return(arcmin)
    '''
    return diameter/distance/1000/np.pi*180*60

def conversion(v,bmaj,bmin):
    '''
    v(GHz),bmaj(deg),bmin(deg),retrun(Jy/beam -> K)
    '''
    if bmaj == bmin:
        beam_sigma = bmaj*u.deg
        beam_area = 2*np.pi*(beam_sigma)**2
        freq = v*u.GHz
        equiv = u.brightness_temperature(beam_area, freq)
    else:
        bmaj = bmaj*u.deg
        bmin = bmin*u.deg
        fwhm_to_sigma = 1./(8*np.log(2))**0.5
        beam_area = 2.*np.pi*(bmaj*bmin*fwhm_to_sigma**2)
        freq = v*u.GHz
        equiv = u.brightness_temperature(beam_area, freq)
    return u.Jy.to(u.K, equivalencies=equiv)

def dist(l,b,d,V,v_sun = 240 ,r_sun = 8.34):
    '''
    return Galactic rotation model
    l(deg) b(deg) d(a list of distance) return(v,d)
    '''
    b = np.deg2rad(b)
    l = np.deg2rad(l)
    r = (r_sun**2+(d*np.cos(b))**2-2*r_sun*d*np.cos(b)*np.cos(l))**0.5
    v = V*r_sun*np.sin(l)*np.cos(b)/r-v_sun*np.sin(l)*np.cos(b)
    return v,d

def con(data,head,region,levels):
    '''
    plot contour map
    '''
    l1,l2,b1,b2 = region
    x1,y1,x2,y2 = coo_box(head,region)

    plt.subplots()
    plt.contour(data,levels=levels,origin='lower',interpolation='nearest',extent=[l2,l1,b1,b2])
    plt.colorbar()

    plt.xlim(l2,l1)
    plt.ylim(b1,b2)
    plt.grid()

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

def plot_spec(file,region,on,spec_v):
    '''
    plot the spectrum of a region in a 3D fits file.
    file(the file path),spec_v()
    '''
    spec = fits.open(file)
    spec_head = spec[0].header
    if spec_head['BUNIT'] == 'K' and spec_head['NAXIS'] == 4:
        spec_data = spec[0].data[0,:,:,:]
    elif spec_head['BUNIT'] == 'K' and spec_head['NAXIS'] == 3:
        spec_data = spec[0].data[:,:,:]
    else:
        spec_data = spec[0].data[:,:,:]*conversion(1.4,spec_head['BMAJ'],spec_head['BMIN'])
    spec.close()
    spec_head['CUNIT3'] = 'm/s'

    v = velocity(spec_data,spec_head)

    spec_on,spec_reg  = circle(spec_data,spec_head,1,region,on,spec_v)
    T_on     = np.mean(np.mean(spec_on,axis=1),axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(v,T_on)

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
    return int(x1),int(y1),int(x2),int(y2)

def coo_tri(head,region):
    '''
    get triangle pixel coordinates
    l1,b1,l2,b2,l3,b3 = region
    '''
    l1,b1,l2,b2,l3,b3 = region
    w = wcs.WCS(head)
    if head['NAXIS'] == 2:
        x1,y1 = w.wcs_world2pix(l1,b1,0)
        x2,y2 = w.wcs_world2pix(l2,b2,0)
        x3,y3 = w.wcs_world2pix(l3,b3,0)
    if head['NAXIS'] == 3:
        x1,y1,v = w.wcs_world2pix(l1,b1,0,0)
        x2,y2,v = w.wcs_world2pix(l2,b2,0,0)
        x3,y3,v = w.wcs_world2pix(l3,b3,0,0)
    if head['NAXIS'] == 4:
        x1,y1,v,s = w.wcs_world2pix(l1,b1,0,0,0)
        x2,y2,v,s = w.wcs_world2pix(l2,b2,0,0,0)
        x3,y3,v,s = w.wcs_world2pix(l3,b3,0,0,0)
    return x1,y1,x2,y2,x3,y3

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
    return(data_onoff,data_region)
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


    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    neg = ax.imshow(result1,origin='lower',interpolation='nearest',extent=[l2,l1,b1,b2])
    fig.colorbar(neg,ax=ax)
    ax.add_patch(patches.Rectangle((onoff[0], onoff[2]),onoff[1]-onoff[0],    \
                                   onoff[3]-onoff[2],color='r',fill=False))
    ax.set_xlim(l2,l1)
    ax.set_ylim(b1,b2)

    x1,y1,x2,y2 = coo_box(head,onoff)
    if data.ndim == 2:
        result2 = data[y1:y2,x1:x2]
    if data.ndim == 3:
        result2 = data[:,y1:y2,x1:x2]

    return result2,result0


def tri(data,head,contrast,region,onoff,*args):
    '''
    plot box pixel coordinates
    return(data_onoff,data_region)
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


    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    neg = ax.imshow(result1,origin='lower',interpolation='nearest',extent=[l2,l1,b1,b2])
    fig.colorbar(neg,ax=ax)

    a=np.array(([onoff[0], onoff[1]],[onoff[2],onoff[3]],[onoff[4],onoff[5]]))
    ax.add_patch(patches.Polygon(a,color='r',fill=False))


    x1,y1,x2,y2,x3,y3 = coo_tri(head,onoff)
    tri = np.int64(np.array(([x1,y1],[x2,y2],[x3,y3]))-[np.min((x1,x2,x3)),np.min((y1,y2,y3))])

    def PointInsideTriangle(pt,tri):
        '''checks if point pt(2) is inside triangle tri(3x2). @Developer'''
        a = 1/(-tri[1,1]*tri[2,0]+tri[0,1]*(-tri[1,0]+tri[2,0])+tri[0,0]*(tri[1,1]-tri[2,1])+tri[1,0]*tri[2,1])
        s = a*(tri[2,0]*tri[0,1]-tri[0,0]*tri[2,1]+(tri[2,1]-tri[0,1])*pt[0]+ \
            (tri[0,0]-tri[2,0])*pt[1])
        if s<0:
            return False
        else: t = a*(tri[0,0]*tri[1,1]-tri[1,0]*tri[0,1]+(tri[0,1]-tri[1,1])*pt[0]+ \
                  (tri[1,0]-tri[0,0])*pt[1])
        return ((t>0) and (1-s-t>0))

    if data.ndim == 2:
        result2 = cp.copy(data[np.min((y1,y2,y3)):np.max((y1,y2,y3)),np.min((x1,x2,x3)):np.max((x1,x2,x3))])
        for i in range(result2.shape[0]):
            for j in range(result2.shape[1]):
                if not PointInsideTriangle([j,i],tri):
                    result2[i,j]=0
    if data.ndim == 3:
        result2 = cp.copy(data[:,np.min((y1,y2,y3)):np.max((y1,y2,y3)),np.min((x1,x2,x3)):np.max((x1,x2,x3))])
        for i in range(result2.shape[1]):
            for j in range(result2.shape[2]):
                if not PointInsideTriangle([j,i],tri):
                    result2[:,i,j]=0
    return result2,result0


def circle(data,head,contrast,region,onoff,*args):
    '''
    plot circle pixel coordinates
    return(data_onoff,data_region)
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

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    neg = ax.imshow(result1,origin='lower',interpolation='nearest',extent=[l2,l1,b1,b2])
    fig.colorbar(neg,ax=ax)
    ax.add_patch(patches.Circle((onoff[0], onoff[1]),onoff[2],color='r',fill=False))

    ax.set_xlim(l2,l1)
    ax.set_ylim(b1,b2)

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
