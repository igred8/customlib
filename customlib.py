# ==========
# Created by Ivan Gadjev
# 2020.02.01
#
# Library of custom functions that aid in projects using pyRadia, genesis, elegant, and varius other codes and analyses. 
#   
# 
# ==========

import sys
import os
import time
import bisect

import scipy.constants as pc
import numpy as np
import pandas as pd

import scipy.signal as sps

import matplotlib.pyplot as plt
import matplotlib.cm as mcm

#
# ===== global defs =====
#

xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])
origin = np.array([0,0,0])

# physical constants
mc2 = 1e-6 * (pc.m_e*pc.c**2)/pc.elementary_charge # MeV. electron mass

#
# ===== helper functions =====
#

def rot_mat(angle, dim=3, axis='z'):
    """ Create a rotation matrix for rotation around the specified axis.
    """
    # sine cosine
    c = np.cos(angle)
    s = np.sin(angle)

    if dim == 3:
        if axis == 'x':
            rr = np.array( [[1, 0, 0], 
                            [0, c, -s], 
                            [0, s, c]] )
        elif axis == 'y':
            rr = np.array( [[c, 0, s], 
                            [0, 1, 0], 
                            [-s, 0, c]] )
        elif axis == 'z':
            rr = np.array( [[c, -s, 0], 
                            [s, c, 0], 
                            [0, 0, 1]] )
    elif dim == 2:
        rr = np.array( [[c, -s], 
                        [s, c]] )
    else:
        print(' error: `dim` variable must be 2 or 3. Specifies the dimension of the vectors that the rotation is acting on.')
    return rr

def rgb_color(value, cmapname='viridis'):
    """ returns the RGB values of the color from the colormap specified with scaled value.
    value - between 0 and 1
    cmapname='viridis' name of the colormap
    """ 
    cmap = mcm.ScalarMappable(cmap=cmapname)
    return list(cmap.to_rgba(value, bytes=False, norm=False)[:-1])

def gaussian(xvec, mean, std):
    """ Returns the Gaussian of the xvec normalized so that the peak is unity.
    xvec - ndarray of floats 
    mean - float. the center of the Gaussian
    std - float. the standard deviation of the Gaussian

    yvec = Exp( - (x - mean)^2 / 2std^2 )

    """

    yvec = np.exp( - (xvec - mean)**2 / (2 * std**2) )

    return yvec

def gaussianDx(x, mu, sig):
    """ First derivative of a Gaussian. Normalized so that its maximum is 1.
    x - ndarray
    mu - float. mean
    sig - float. std dev
    
    """
    
    y = (-(x - mu)/sig**2) * gaussian(x, mu, sig)
    # NOTE: normalization to maximum value = 1
    y = y/np.abs(y).max()
    
    return y

def gaussianDx2(x, mu, sig):
    """ Second derivative of a Gaussian (inverted sombrero). Normalized so that its maximum is 1.
    x - ndarray
    mu - float. mean
    sig - float. std dev
    
    """
    
    y = ( -1/sig**2 + ((x - mu)/sig**2)**2 ) * gaussian(x, mu, sig)
    # NOTE: normalization to maximum value = 1
    y = y/np.abs(y).max()
    return y

def stepfunc(xvec, center, width):
    """
    Returns a step function with center and total width.
    xvec - ndarray of floats 
    center - float
    width - float. total width of box or step
    """

    yvec = [ int((xx >= center - width/2) and (xx <= center + width/2)) for xx in xvec]

    return yvec

def smooth(xvec, yvec, width, mode='gauss',edgemode='cut'):
    """
    xvec
    yvec
    
    width - float. if mode == 'gauss' width=std, if 'step' [center-width/2, center+width/2]
    mode - {'gauss','step', 'median'}
    edgemode = {'cut','pad'}
    to be used with gaussian or stepfunc
    """
    center = xvec.min() + (xvec.max() - xvec.min() ) / 2
    xstep = xvec[1]-xvec[0]
    xlen = xvec.shape[0]

    if mode in ['gauss', 'step']:
        if mode == 'gauss':
            nsig = 5
            xvectemp = np.arange(center - nsig * width, center + nsig * width + xstep, xstep)
            wvec = gaussian(xvectemp, center, width)
        elif mode == 'step':
            xvectemp = np.arange(center - width/2, center + width/2 + xstep, xstep)
            wvec = stepfunc(xvectemp, center, width)
        
        cnorm = 1 / (np.sum(wvec))
        xvec = cnorm * np.convolve(xvec, wvec, mode='valid')
        yvec = cnorm * np.convolve(yvec, wvec, mode='valid')
    
    elif mode == 'median':
        yvec = sps.medfilt(yvec, kernel_size=width)
    else:
        print("ERROR: _mode_ must be 'gauss', 'step', or 'median' ")
        return 1
    
    ndiff = xlen - xvec.shape[0]
    if ndiff < 0:
        print('WARNING: `width` is larger than `xvec` span.')
    else:
        if edgemode == 'pad':
            xvec = np.pad(xvec, [ndiff // 2, ndiff - ndiff // 2], mode='edge')
            yvec = np.pad(yvec, [ndiff // 2, ndiff - ndiff // 2], mode='edge')
        

   
    return xvec, yvec

#
# === Integrate

def numint(xvec, yvec):

    """ Calculates the integral of the function defined by f(xvec) = yvec
    xvec - ndarray 
    yvec - ndarray
    
    """
    # differential element vector
    delxvec = np.abs(xvec[1:] - xvec[:-1])
    delxvec = np.append(delxvec, delxvec[-1]) # make same length
    
    numeric_integral = np.sum( delxvec * yvec )
    return numeric_integral


#
# === physics specific functions
#

def bwlimit(freq0, timeFWHM, mode='gauss'):
    """ Calculate FWHM BW of a laser pulse, given a central frequency and the pulse length FWHM.

    return freqFWHM, wlFWHM
    """

    bwfactor = {'unit':1.0, 'gauss':0.441, 'sech':0.315}
    freqFWHM = bwfactor[mode] / timeFWHM

    # convert to wavelength
    wlFWHM = (pc.c / freq0**2) * freqFWHM

    return freqFWHM, wlFWHM

#
# === beam dynamics

def gamma_beta_rel(totengMeV, restmassMeV):
    """ returns the relativistic factors gamma and beta for a given particle energy (gmc2) and rest mass (mc2)

    gamma = totengMeV / restmassMeV
    beta = sqrt( 1 - 1/g^2)
    """
    gamma = totengMeV / restmassMeV
    beta = np.sqrt( 1 - 1 / gamma**2)

    return gamma, beta

def mom_rel(totengMeV, restmassMeV):
    """ returns the relativistic momentum (in units of MeV/c) of the particle with the specified total energy (gmc2) and rest mass (mc2) 

    pc = sqrt( E^2 - mc2^2 )

    NB: the output is in units of MeV/c. 
    """
    prel = np.sqrt( totengMeV**2 - restmassMeV**2)
    return prel

def mom_rigid(momMeV, charge):
    """ Returns the momentum rigidity for a given momentum in MeV and charge in units of elementary charge units, e. 
    
    B * rho = (1e6 / c) * (1 / charge) * pc 
    """
    br = (1e6 / pc.c) * (momMeV / charge)

    return br

def emit_convert(emit_old, frac_old, frac_new):
    """ Convert to emittance that encompases the given fraction.

    emit_old - float. geometric or normalized emittance that encompases `frac_old` of the beam
    frac_old - float. [0,1] the fraction of the beam encompassed by the emit_old 
    frac_new - float. [0,1] new fraction encompased by emittance

    This is just a factor that multiplies the input emittance. 
    """

    emit_new = emit_old * (np.log(1 - frac_new) / np.log(1 - frac_old))

    return emit_new

## Transfer matrix formalism

def mat_quad(maglen, gradient, totengMeV, restmassMeV, charge=1, focus='focus'):
    """ the 2x2 1st order matrix for quadrupole focusing.

    1/f = Kquad * L
    f - focal length
    Kquad = gradient / momentumrigidity # this is the S.Y.Lee definition (note this is not squared!)
    L - magnetic length
    """
    mrig = mom_rigid(mom_rel(totengMeV,restmassMeV), charge)
    kquad = gradient / mrig # normalized quadrupole gradient
    if focus == 'focus':
        fsign = -1
    else:
        fsign = 1

    fl = fsign / (kquad * maglen)

    mat = np.array( [[ 1, 0 ],
                     [ 1/fl, 1 ]
                     ])
    return mat

def mat_drift(dlen):
    """ return the 2x2 matrix for a drift
    """
    mat = np.array( [[ 1, dlen ],
                     [ 0, 1 ]
                     ])
    return mat

def make_M_mat(mat):
    """ returns the 3x3 matrix that acts on [beta,alpha,gamma] from the 2x2 matrix for [x,x']
    mat - ndarray with shape [2,2]
    """
    m11 = mat[1,1]
    m12 = mat[1,2]
    m21 = mat[2,1]
    m22 = mat[2,2]

    mat3 = np.array( [[ m11**2, -2*m11*m12, m12**2 ],
                      [ -m11*m21, m11*m22 + m12*m21, -m12*m22 ],
                      [ m21**2, -2*m21*m22, m22**2 ]
                     ])
    return mat3

#
# === ICS analytic formulas

def eng_photon(wavelen):
    """ Returns the energy of the photon with the given wavelength.

    wavelen - float. meter.

    returns the energy in MeV
    """
    hbark = 1e-6 * pc.hbar * pc.c * 2 * pc.pi / wavelen / pc.e

    return hbark

def ics_gammaCM(engele, englas, angin):
    """ The relativistic gamma factor of the center of momentum frame.
    engele - float. MeV energy of incoming electron
    englas - float MeV energy of incoming laser photon
    angin - float. radians angle between the incoming electron and photon in the lab frame


    """
    x =  (4 * engele * englas) / mc2**2

    gcm = ( engele + englas ) / ( mc2 * np.sqrt(1 + (x/2) * (1 - np.cos(angin)) ) )

    return gcm

def ics_betaCM(engele, englas, angin):
    """ The beta factor of the center of momentum frame.
    engele - float. MeV energy of incoming electron
    englas - float MeV energy of incoming laser photon
    angin - float. radians angle between the incoming electron and photon in the lab frame

    """
    bcm = np.sqrt( 1 - ( 2*engele*englas*(1 - np.cos(angin)) ) / ( engele + englas )**2 )

    return bcm

def ics_eng_ph(engele, englas, angin, angout):

    """ Calculate the energy of the scattered photon in the outgoing angle.
    This is the full formula when eqs 6 and 11 of Curatolo et al. (2017) are combined.
    
    engele - float. MeV energy of incoming electron
    englas - float MeV energy of incoming laser photon
    angin - float. radians angle between the incoming electron and photon in the lab frame
    
    returns engph in MeV
    """
    gcm = ics_gammaCM(engele, englas, angin)
    bcm = ics_betaCM(engele, englas, angin)

    angoutdep = ( 1 + bcm*np.cos(angout) ) / ( np.cos(angout)**2 + gcm**2 * np.sin(angout)**2 )

    engph = englas * gcm**2 * (1 - np.cos(angin)) * angoutdep

    return engph

def ics_sig(eng_ele, eng_las):
    """ Calculates the Compton scattering cross-section as a proportion of the Thomson cross section.
    sigma_T = 8*pi*re^2/3

    eng_ele - float. MeV. total energy of the incoming electron
    eng_las - float. MeV. total energy of the incoming laser photons
    
    x - unitless. defined as: x = 4*E_e*E_L / (m_e*c^2)^2. It is a measure of the recoil effect due to the high electron energy or photon energy when compared to the rest mass of the electron.

    returns sigma/sigma_T

    !!! NB: if x is too small, numerical erros due to division by small numbers and subtraction of small numbers begin to take over. usually around x<1e-4. In small x limit, sigma/sigma_T is well approximated by simply (1-x).
    """
    x = 4 * eng_ele * eng_las / mc2**2
    
    sigbysigT = ((3/4)
              *(1/x)
              *(1/2 + 8/x - 1/(2*(1+x)**2) + (1 - 4/x - 8/x**2)*np.log(1 + x) )
              )
    return sigbysigT

def ics_lum_headon(eng_ele, eng_las, num_ele, num_las, sigma_ele_x, sigma_ele_y, sigma_las, reprate=1.0):
    """ Calculate the luminosity for a HEAD-ON collision between Gaussian beams. 
    eng_ele - float. MeV. total energy of the incoming electron
    eng_las - float. MeV. total energy of the incoming laser photons
    num_ele - float. number of electrons
    num_las - float. number of laser photons
    sigma_ele_x, sigma_ele_y - float. meter. x and y RMS of electron beam
    sigma_las - float. meter. sigma_las = waist/2
    reprate - float. number of beam-beam collisions per second.

    luminosity =  N_e * N_l * reprate * cross_section / overlap
    """

    overlap = 2*pc.pi*np.sqrt(sigma_ele_x**2 + sigma_las**2)*np.sqrt(sigma_ele_y**2 + sigma_las**2)

    sigThomson = pc.physical_constants['Thomson cross section'][0]
    sigTotal = sigThomson * ics_sig(eng_ele, eng_las)
    lum = num_ele * num_las * reprate  * sigTotal/ overlap

    return lum


#
##
### === Transfer matrix beamlines
##
#

class ElementABCD(object):
    """
    Optical element represented by an ABCD matrix. 

    drift:
        properties = {'position':0,
                              'length':0}
    lens:
        properties = {'position':0,
                              'focal_len':0}
    """
    def __init__(self, name, eletype='drift', eleprops={'position':0,'length':0}):
        
        self.name = name
        self.element_type = eletype
        self.properties = eleprops
        
        if self.element_type == 'drift':
            self.abcd_mat = np.array([[1, self.properties['length']],
                                      [0, 1]])
        
        elif self.element_type == 'lens':
            self.abcd_mat = np.array([[1, 0],
                                      [-1 / self.properties['focal_len'], 1]])
        
        else:
            print('WARNING: Element type not supported. Created drift with zero length.')

        

class BeamLine(object):
    """
    Optical beamline class. 
    Store elements as matricies (ABCD or electron transfer matricies). 

    """

    def __init__(self):
        
        self.element_list = []
        self.element_names = []
        self.element_position = []

    def add_element(self, element):
        """ Adds an element class to the beamline.
        
        element is a class ElementABCD() or similar.
        """
        try:
            # insert the element based on its positionin along the beamline
            posi = element.properties['position']
            bisect.insort(self.element_position, posi)
            addindex = self.element_position.index(posi)
            self.element_list.insert(addindex, element)
            self.element_names.insert(addindex, element.name)
            
        except AttributeError:
            print('ERROR! element does not possess the required attributes. Need class with .name and .properties()')
            return 1

        return 0

    def del_element(self, elementname):
        """ Deletes the specified element from the beamline
        """
        try:
            delindex = self.element_names.index(elementname)
            del self.element_list[delindex]
            del self.element_names[delindex]
            del self.element_position[delindex]

        except ValueError:
            print('WARNING: Beamline does not contain element with that name.')
            return 1

        return 0

    def make_mat(self, senter, sexit):
        """ Based on the given location in the beamline, find the total transfer matrix.
        Assumes that the elements are sorted by the sort_element() routine.
        """
        # init transport matrix
        transportmatrix = np.eye(2)
        # print(transportmatrix)

        # position temporary
        si = senter
        
        # include elements up to sexit point
        s0index = bisect.bisect(self.element_position, senter) 
        s1index = bisect.bisect(self.element_position, sexit)
        # print(s0index)
        # print(s1index)
        for i,ele in enumerate(self.element_list[s0index:s1index],start=s0index):
            # print(ele.name)
            # drift to element from previous position
            driftlen = self.element_position[i] - si
            # print(driftlen)
            # print(transportmatrix)
            matdrift = ElementABCD('tempdrift', eletype='drift', eleprops={'position':0, 'length':driftlen}).abcd_mat
            # print(matdrift)
            transportmatrix = np.matmul(matdrift, transportmatrix)
            # print(transportmatrix)
            # element
            transportmatrix = np.matmul(ele.abcd_mat, transportmatrix)
            # print(transportmatrix)
            try:
                # sold = si
                si = self.element_position[i]
            except IndexError:
                
                continue

        

        # drift to final position from last element before final position
        driftlen = sexit - si
        
        matdrift = ElementABCD('tempdrift', eletype='drift', eleprops={'position':0, 'length':driftlen}).abcd_mat

        transportmatrix = np.matmul(matdrift, transportmatrix)
        # print(transportmatrix)


        return transportmatrix 

    def ray_trace(self, invec, inpos, outpos):
        """ Given a set of input vectors with initial transverse postion and angle, calculate their transport through the beamline at teh specified outpos.

        invec - ndarray.shape = (2,n)
        inpos - float position at which invec is specified
        outpos = ndarray.shape = (m,)

        """
        # inital transport to first point
        transmat = self.make_mat(inpos, outpos[0])
        outvec = np.matmul(transmat, invec)
        outvec = np.stack((invec, outvec))

        for i,ss in enumerate(outpos[1:], start=1):

            transmat = self.make_mat(outpos[i-1], ss)
            outvectemp = np.matmul(transmat, outvec[-1])
            outvec = np.concatenate((outvec, [outvectemp]), axis=0)

        outpos = np.insert(outpos, 0, inpos)

        return outpos, outvec
            

    


#
##
### === Particle Swarm Optimization
##
#

class PSO(object):
    """ Class handles Particle Swarm Optimization (PSO) of the magnet dimensions with a target gradient, strenght, magnetic length, etc.

    Particle Swarm Optimization pseudo code:
    - define seach space, SS
    - init particles inside SS
    - init particle velocity
    - init global best solution, gbest
        - compute cgbest = cost(f(gbest))
    while (termination condition not met):
        - init particle best solution, pbest
            - compute cpbest = cost(f(pbest))
        for particle in particles:
            
            - compute cost
                cparticle = cost(f(particle))
                
            - update gbest and pbest:
                if cpbest > cparticle:
                    pbest = particle
                    cpbest = cost(f(pbest))
                    if cgbest > cpbest:
                        gbest = pbest
                        cgbest = cpbest
                
            - update velocity
                v = v + alpha1*rand(0,1)*(pbest - particle) + alpha2*rand(0,1)*(gbest - particle)
            - update position
                particle = particle + v

    """
    def __init__(self):

        self.phi1 = 2.05
        self.phi2 = 2.05

        self.maxiter = 0
        self.precision = 0

    def cost(self, current, target):
        """ Calculates the total square difference between the current parameter values and the target values.

        current - np.array(n,)
        target - np.array(n,)
        """

        value = np.sqrt( np.sum( (current - target)**2 ))
        
        return value

    def velocity(self, vin, xin, pbest, gbest):
        """

        Updates the input velocity.
        vin - np.array(nparticles, ) input velocity
        xin - np.array(nparticles, ) input position
        pbest - np.array(nparticles, ) the best position for the particle so far
        gbest - np.float the best position for any particle 
        phi1=2.05, phi2=2.05 - regulate the 'randomness' in the velocity update as described below
        
        Clerc and Kennedy (2002) noted that there can be many ways to implement the constriction coefficient. One of the simplest methods of incorporating it is the following :
        v_(i+1) = chi * ( v_i + U(0,phi1) * (p_i - x_i) + U(0,phi2) * (pg - x_i) )
        x_(i+1) = x_i + v_i
        where,
        phi = phi1 + phi2 > 4
        chi = 2 / ( phi - 2 + sqrt(phi^2 - 4*phi) )

        When Clerc's constriction method is used, phi is commonly set to 4.1, phi1=phi2 and the constant multiplier chi is approximately 0.7298. This results in the previous velocity being multiploied by 0.7298 and each of the two (p - x) terms being multiplied by a random number limited by 0.7398*2.05 = 1.49618.
        """
        phi = self.phi1 + self.phi2
        chi = 2 / ( phi - 2 + np.sqrt(phi**2 - 4 * phi) )

        vout = chi * ( vin + self.phi1 * np.random.random(xin.shape) * (pbest - xin) + self.phi2 * np.random.random(xin.shape) * (gbest - xin) )

        return vout

    def run_pso(self, function, searchspace, target, nparticles, maxiter, precision, domain, verbose=True):

        """ Performs a PSO for the given function in the searchspace, looking for the target, which is in the output space.
        
        function - the function to be optimized. Its domain must include the seachspace and its output must be in the space of target.
        searchspace - np.array((ssdim, 2)) 
        target - np.array((tdim, ))
        nparticles - number of particles to use in the optimization
        maxiter - maximum number of iterations to the optimization routine
        precision - how close to the target to attemp to get
        domain - absolute boundaries on the trial solutions/particles
        """
        # update attributes
        self.maxiter = maxiter
        self.precision = precision

        # search space dimensionality
        if searchspace.shape[1] != 2:
            print('WARNING! searchspace does not have dimenstions (N,2).')
        ssdim = searchspace.shape[0]
        
        # init particle positions and velocities
        xpart = np.random.random((nparticles, ssdim))
        for ii in range(ssdim):
            xpart[:,ii] = (searchspace[ii,1] - searchspace[ii,0]) * xpart[:,ii] + searchspace[ii,0] # scale the uniform radnom dist
        
        vpart = np.zeros(xpart.shape)

        # init particle best solution
        pbest = 1.0 * xpart
        cpbest = np.array([ self.cost(function(*xp), target) for xp in pbest ])
        # init global best solutions
        im = np.argmin(cpbest)
        gbest = pbest[im]
        cgbest = cpbest[im]

        if False:
            return xpart, vpart, pbest, cpbest, gbest, cgbest

        # intermediate arrays
        # multiply by 1.0 to make copies not bind references
        xarr = 1.0 * xpart[:,:,None]
        varr = 1.0 * vpart[:,:,None]
        parr = 1.0 * pbest[:,:,None]
        cparr = 1.0 * cpbest[:,None]
        garr = 1.0 * gbest[:,None]
        cgarr = 1.0 * np.array([cgbest])

        iternum = 0

        t1 = time.time()
        while (iternum <= maxiter) and ( cgbest > precision):
        # while (iternum <= maxiter):

            for pp in range(nparticles):
                
                
                # update velocity
                vpart[pp] = self.velocity(vpart[pp], xpart[pp], pbest[pp], gbest)
                # update position
                xpart[pp] = xpart[pp] + vpart[pp]
                
                # keeps particles inside the absolute boundaries given by `domain`
                xpart[pp] = np.maximum(xpart[pp], domain[:,0])
                xpart[pp] = np.minimum(xpart[pp], domain[:,1])

                # compute cost of new position
                cpp = self.cost(function(*xpart[pp]) , target )

                # update best position
                if cpp < cpbest[pp]:
                    pbest[pp] = xpart[pp]
                    cpbest[pp] = cpp
                if cpp < cgbest:
                    gbest = xpart[pp]
                    cgbest = cpp

            xarr = np.concatenate((xarr, xpart[:,:,None]),axis=2)
            varr = np.concatenate((varr, vpart[:,:,None]), axis=2)
            parr = np.concatenate((parr, pbest[:,:,None]), axis=2)
            cparr = np.concatenate((cparr, cpbest[:,None]), axis=1)
            garr = np.concatenate((garr, gbest[:,None]), axis=1)
            cgarr = np.append(cgarr, cgbest)

            iternum += 1

        t2 = time.time()
        if verbose:
            print('optimization took {:5.2f} seconds'.format(*[t2-t1]))

        return xarr, varr, parr, cparr, garr, cgarr


