"""
Created on Wed Feb  2 13:04:21 2022

@author: Chait
"""

import numpy as np
import matplotlib.pylab as plt
from scipy import optimize 
from sklearn.linear_model import LinearRegression


# Define global physical constants
Avogadro = 6.02214086e23
Boltzmann = 1.38064852e-23
MSD = []


def PeriodicBoundaryConditions(pos, box):
    
    ndims = len(box)

    for i in range(ndims):
        #vels[((pos[:,i] <= box[i][0]) | (pos[:,i] >= box[i][1])),i] *= -1
        if i == 2:
            break
        
        elif np.all(pos[:, i] <= box[i][0]):
            pos[:, i] = box[i][1]
        elif np.all(pos[:, i] >= box[i][1]):
            pos[:, i] = box[i][0]
        

def TimeEvolution(pos, vels, forces, mass,  dt):
    
    
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T
    
def ForceCalculation(mass, pos, box, vels, temp, relax, dt):
    

    natoms, ndims = vels.shape

    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T
    f1 = - (vels * mass[np.newaxis].T)
    f2 = wallforce(ff, pos, box)
    force = f1 / relax + noise + f2
    
    
    return force


def f_x(x, a, b):
    return a*np.power(x, -12) + b


def forcefield():
    file1 = open("C:\\Users\\Chait\Desktop\\Langevin Dynamics\\x_data_wp.txt", 'r')
    data1 = np.loadtxt(file1, skiprows = 1)*1e-9
    #print(data1)

    file2 = open("C:\\Users\\Chait\Desktop\\Langevin Dynamics\\E_data_wp.txt", 'r')
    data2 = np.loadtxt(file2, skiprows = 1)
    #print(data2)
    
    ff = -np.gradient(data2, data1)
    
    forcefield.d1 = data1
    forcefield.d2 = data2
    
    x = data1[0:10]
    #print(x)
    y = data2[0:10]
    #print(y)
    
    popt, pcov = optimize.curve_fit(f_x, x, y)
    #print( "this is popt :", popt)
    forcefield.a = popt[0]
    forcefield.b = popt[1]
    
    return ff


def wallforce(ff, pos, box):
    #global fce
    cutoffpnt = 5.3e-9
    nearestpnt = forcefield.d1[0]
    ndims = len(box)
    temporpst = np.copy(pos)
    temporpst = temporpst.tolist()
    fff = []
    
    for count, pst in enumerate(temporpst):
        h = []
        
        for i in range(ndims):
            if i == 0 or i == 1:
                h.append(0)
                continue
            
            if pst[i] - box[i][0] < cutoffpnt:
                if pst[i] - box[i][0] < nearestpnt:
                    fce = 12*forcefield.a*((pst[i] - box[i][0]))**-13
                else:
                    diffinpst = np.abs(forcefield.d1[:] - (pst[i] - box[i][0]))
                    clst_index = diffinpst.argmin()
                    fce = ff[clst_index]
                h.append(fce)
            elif box[i][1] - pst[i] < cutoffpnt:
                if box[i][1] - pst[i] < nearestpnt:
                    fce = 12*forcefield.a*((box[i][1] - pst[i]))**-13
                else:
                    diffinpst = np.abs(forcefield.d1[:] - (box[i][1] - pst[i]))
                    clst_index = diffinpst.argmin()
                    fce = ff[clst_index]
                    fce *= -1
                h.append(fce)
            else:
                h.append(0)
        fff.append(h)
    fff = np.asarray(fff)
    return fff


def gz(pos, box):
    natoms = 1000
    disinz = box[2][1] - box[2][0]
    temppst = np.copy(pos)
    pst_new = temppst[:, 2]
    pst_new = pst_new.tolist()
    g_z = []
    p_z = []

    width = disinz/66
    s = []
    for i in np.arange(0, 11e-8, width):
        s.append(i)
    s_new  = s[0:67]
    
    for i in range(len(s_new)-1):
        g = []
        cnt = 0
        for count, pst in enumerate(pst_new):
            
            if pst >= s_new[i] and pst < s_new[i+1]:
                g.append(pst)
                cnt += 1
        rto = cnt/natoms
        cnt = 0
        g = 0
        p_z.append(g)
        g_z.append(rto) 
    g_z = np.asarray(g_z)
    p_z = np.asarray(p_z)
    gz.pd = g_z
    return gz

def PartitionFunction(pos, box):
    distinz = box[2][1] - box[2][0]
    nstpnt = forcefield.d1[0]
    ctpnt = 5.3e-9 
    T = 293.15
    ke = (Boltzmann*T)/2
    bheta = 1/(Boltzmann*T)
    tpst = np.copy(pos)
    tpst_new = tpst[:, 2]
    tpst_new = tpst_new.tolist()
    p = []
    k = []
    z = []
    ee = []
    wdth = distinz/66
    sl = []
    for i in np.arange(0, 11e-8, wdth):
        sl.append(i)
    sl_new = sl[0:67]

    for i in range(len(sl_new) - 1):
        j = []
        cnt = 0
        for count, pst in enumerate(tpst_new):
        
            if pst - box[2][0] >= sl_new[i] and pst - box[2][0] < sl_new[i+1]:
                if pst - box[2][0] < nstpnt:
                    pe = (forcefield.a*(pst - box[2][0])**-12) + forcefield.b
                    cnt += 1
                elif pst - box[2][0] >= nstpnt and pst - box[2][0] < ctpnt:
                    dpst = np.abs(forcefield.d1[:] - (pst - box[2][0]))
                    c_index = dpst.argmin()
                    pe = forcefield.d2[c_index]
                    cnt += 1
                else:
                    pe = 0
                    cnt += 1
                j.append(pe)
            elif box[2][1] - pst >= sl_new[i] and box[2][1] - pst < sl_new[i+1]:
                if box[2][1] - pst < nstpnt:
                    pe = (forcefield.a*(box[2][1] - pst)**-12) + forcefield.b
                    cnt += 1
                elif box[2][1] - pst >= nstpnt and box[2][1] - pst < ctpnt:
                    dpst = np.abs(forcefield.d1[:] - (box[2][1] - pst))
                    c_index = dpst.argmin()
                    pe = forcefield.d2[c_index]
                    pe *= -1
                    cnt += 1
                else:
                    pe = 0
                    cnt += 1
                j.append(pe)
            else:
                continue
                    
        s_pe = sum(j)
        a_pe = s_pe/cnt
        eng = a_pe + ke
        pd = bheta*eng
        z_i = np.exp(-pd)
        cnt = 0
        j *= 0
        ee.append(eng)
        z.append(z_i)
        k.append(j)
        
    z_sum = sum(z)
    
    
    for i in np.arange(len(sl_new) - 1):
        prb = (1/z_sum)*z[i]
        
       
    p.append(prb)
    return p
                
                

def run(**args):
    

    natoms, box, dt, temp = args['natoms'], args['box'], args['dt'], args['temp']
    mass, relax, nsteps   = args['mass'], args['relax'], args['steps']
    freq, radius =  args['freq'], args['radius']
    
    dim = len(box)
    pos = np.random.rand(natoms,dim)

    for i in range(dim):
        pos[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * pos[:,i]

    vels = np.random.rand(natoms,dim)
    mass = np.ones(natoms) * mass / Avogadro
    radius = np.ones(natoms) * radius
    step = 0

    output = []
    c = []
    e = []
    run.p = pos

    #PartitionFunction(pos, box)
    
    gz(pos, box)
    
    pos = np.sort(pos)
    new_pos = np.copy(pos)
    
   
                
    while step <= nsteps:

        step += 1
        

        # Compute all forces
        forces = ForceCalculation(mass, pos, box, vels, temp, relax, dt)

        # Move the system in time
        TimeEvolution(pos, vels, forces, mass, dt)

        # Applying the periodic boundary conditions
        PeriodicBoundaryConditions(pos, box)
        
        if not step%freq:
            for i in range(natoms):
            
                difference = pos[i, 2] - new_pos[i, 2]
                #print(difference)
                sqd = np.square(difference)
            
                #print(sqd)
                new_pos = np.copy(pos)
                #m = np.mean(sqd)
                c.append(sqd.mean())
                dfc = sqd.mean()/(2*value*dt)
                e.append(dfc)
            #print(cv)
    c = np.asarray(c)
    e = np.asarray(e)
          
            
    #print(e)
    return e.mean()

if __name__ == '__main__':
    D = []
    ff = forcefield()
    time = range(150,300,20)
    for count,value in enumerate(time):
        params = {
            'natoms': 1000,
            'temp': 293.15,
            'mass': 0.00072066,
            'radius': 5.2e-10,
            'relax': 8.8e-16,
            'dt': 1e-16,
            'steps': 10000,
            'freq': value,
            'box': ((0, 1e-8), (0, 1e-8), (0, 1e-8)),
        }

        output = run(**params)
        D.append(output)
        
    
    time = np.asarray(time)
    time = time*1e-16
    time.tolist()
    
    lr = LinearRegression()
    lr.fit(time.reshape(-1,1), D)
    A_pred = lr.predict(time.reshape((-1,1)))
    print(lr.coef_)
    print(lr.intercept_)
    
    box = ((0, 1e-8), (0, 1e-8), (0, 1e-8))
    s = []
    disinz = box[2][1] - box[2][0]
    width = disinz/66
    for i in np.arange(0, 11e-8, width):
        s.append(i)
    s_new  = s[0:66]
    
    
    
    plt.figure()
    plt.plot(s_new, D)
    plt.xlabel('Distance in Z - axis (m)')
    plt.ylabel(' Diffusion coefficient (m^2/s)')
    plt.show()
    
    plt.figure()
    plt.hist(run.p, bins = s_new, edgecolor = 'black', density = False )
    plt.xlabel('Distance in Z - axis (m)')
    plt.ylabel('frequency')
    plt.show()
    
    plt.figure()
    plt.plot(s_new, gz.pd)
    plt.xlabel('Distance in Z axis')
    plt.ylabel('Probability densities')
    plt.show()
    
    
    