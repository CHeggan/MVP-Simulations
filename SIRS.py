import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.patches as mpatches
from matplotlib.colors import from_levels_and_colors
from scipy.ndimage.measurements import center_of_mass

from latticeGenerator import Lattice

"""Notes:
        - N = n^2
        - choose p1, p2, p3
        - 0 = susceptible, 1 = infected, 2 = Recovered, 3 = Permo Immune
"""

class Simulation:
    def __init__(self, lat):
        self.lat = lat
        self.data = []

def nnCoords(lat, i, j):
    """Finds the coordinates of the NNs:

        Takes:
            n(int)(global)= dimension of array, expect shape of lattice(n, n)
            lat(array(n, n)) = the simulation lattice
            i(int) = column index of point
            j(int) = row index of point

        Returns:
            l, r, u, d (size 2 arrays) = The coordinates of the NNs
            siteData (size 4 array) = the values for NNs: l, r, u, d
    """
    l = [i, (j + 1)%n]
    r = [i, (j - 1)%n]
    u = [(i - 1)%n, j]
    d = [(i + 1)%n, j]

    #Stores all of the actual spins in an array for easier use outside function
    siteData = [ lat[(i - 1)%n, j], lat[(i + 1)%n, j], lat[i, (j + 1)%n],
                lat[i, (j - 1)%n] ]
    return l, r, u, d, siteData

def animate(i, im, sim):
    """The animate function that is sent to func animation:

        Takes:
            lat(array(n, n)) = the simulation lattice
            im(matplotlib object) = the plottng screen, an imshow object

        Returns:
            [im] = updated imshow object for animation to use
    """
    new_lat = sweep(sim.lat)
    sim.lat = new_lat
    im.set_array(new_lat)
    whitePatch = mpatches.Patch(color='teal', label='R')
    blackPatch = mpatches.Patch(color='purple', label='S')
    orangePatch = mpatches.Patch(color='yellow', label='I')
    legend = plt.legend(handles=[blackPatch,orangePatch, whitePatch])
    return [im] + [legend]

###################################
def oneSample(lat):
    [x, y] = np.random.randint(0, high = n, size = 2)
    #if susceptible
    if lat[x, y] == 0:
        l, r, u, d, siteData = nnCoords(lat, x, y)
        if 1 in siteData:
            ranNum = np.random.uniform()
            if ranNum < p1:
                lat[x, y] = 1
                return lat
            else:
                return lat
        else:
            return lat

    #if infected
    if lat[x, y] == 1:
        ranNum = np.random.uniform()
        if ranNum < p2:
            lat[x, y] = 2
            return lat
        else:
            return lat

    #if recovered
    if lat[x, y] == 2:
        ranNum = np.random.uniform()
        if ranNum < p3:
            lat[x, y] = 0
            return lat
        else:
            return lat

    if lat[x, y] == 3:
        return lat

def sweep(lat):
    for i in range(n**2):
        lat = oneSample(lat)
    return lat

def countTypes(lat):
    s = np.count_nonzero(lat == 0)
    i = np.count_nonzero(lat == 1)
    r = np.count_nonzero(lat == 2)
    return s, i, r

def avgInfected(lat):
    i = np.count_nonzero(lat == 1)
    psi = float(i/n**2)
    return psi

def varInfected(infList):
    var = np.var(infList)
    return float(var/len(infList))

def jacknife(vals):
    """Calculates the errors using jacknife method

        Takes:
            vals(array) = data to re-sample

        Returns:
            err(float) = error value on the data
    """
    errList = []
    #Go over the lists and remove ith element at a time, so new array is always size len - 1
    for i in range(len(vals)):
        #remove ith element
        newVals = np.delete(vals, i)

        nextErr = np.var(newVals)/(len(newVals))

        #new lists should end up same length as eVals and mVals
        errList.append(nextErr)

    #The errors on heat capcity and chi are sqrt(size of array)*standard deviation without sqrt(1/m) factor
    error = math.sqrt(len(vals))*math.sqrt(np.var(errList))
    return error


def permoImmune(lat, fraction):
    numImmune = int(fraction* n**2)
    while numImmune != 0:
        [x, y] = np.random.randint(0, high = n, size = 2)
        if lat[x, y] != 3:
            lat[x, y] = 3
            numImmune -= 1
    return lat

def withoutImmune(p1Lims, p3Lims, interval, totSweeps, equil, spacing):
    global p1, p2, p3
    p2 = 0.5
    p1s = np.arange(p1Lims[0], p1Lims[1], interval)
    p3s = np.arange(p3Lims[0], p3Lims[1], interval)

    avgs = np.zeros( shape = (len(p1s), len(p3s)) )
    vars = np.zeros( shape = (len(p1s), len(p3s)) )
    errors = []

    for i in range(len(p1s)):
        for j in range(len(p3s)):
            p1, p3 = p1s[i], p3s[j]

            print('p1: {:.2f}, p2: {:.2f}'.format(p1, p3))

            latticeClass = Lattice(n, 3)
            lat = latticeClass.random

            avgList = []

            for k in range(equil):
                    lat = sweep(lat)
            for m in range(totSweeps):
                lat = sweep(lat)
                if m%spacing == 0:
                    avgList.append(avgInfected(lat))
                if avgInfected == 0:
                    break

            totalAvgI = np.mean(avgList)
            totalVarI = varInfected(avgList)
            err = jacknife(avgList)
            avgs[i][j] = totalAvgI
            vars[i][j] = totalVarI
            errors.append(err)

    return p1s, p3s, avgs, vars, errors

def withImmune(totSweeps, equil, spacing, fraction):
    latticeClass = Lattice(n, 3)
    lat = latticeClass.random
    lat = permoImmune(lat, fraction)

    avgList = []
    for k in range(equil):
            lat = sweep(lat)
    for m in range(totSweeps):
        lat = sweep(lat)
        if m%spacing == 0:
            avgList.append(avgInfected(lat))

        if avgInfected == 0:
            break

    totalAvgI = np.mean(avgList)
    totalVarI = varInfected(avgList)
    return totalAvgI, totalVarI

def runFullSim():
    title1 = 'Countour Plot of Avg Infected Sites - Full'
    title2 = 'Contour Plot of Infected Site Variances - Full'
    p1s, p3s, avgs, vars, errors = withoutImmune([0, 1], [0, 1], 0.05, 1000, 100, 1)

    np.savetxt('fullSimAvgs.txt', avgs)
    np.savetxt('fullSimVars.txt', vars)

    plt.contourf(p1s, p3s,  avgs)
    plt.xlabel('p1')
    plt.ylabel('p3')
    plt.title(title1)
    plt.savefig(title1)
    plt.show()

    plt.contourf(p1s, p3s,  vars)
    plt.xlabel('p1')
    plt.ylabel('p3')
    plt.title(title2)
    plt.savefig(title2)
    plt.show()


def runZoomedSim():
    title = 'Plot of Infected Site Variances - Zoomed'
    p3 = 0.5
    p1s, p3s, avgs, vars, errors = withoutImmune([0.2, 0.5], [0.5, 0.53], 0.05, 10000, 100, 1)

    np.savetxt('slicedSimAvgs.txt', avgs)

    plt.errorbar(p1s, vars, yerr = errors)
    plt.xlabel('p1')
    plt.ylabel('Variance')
    plt.title(title)
    plt.savefig(title)
    plt.show()

def runImmunitySim(numRuns = 3):
    global p1, p2, p3
    p1, p2, p3 = 0.5, 0.5, 0.5

    fullAvgs = []
    errors = []
    newAvgs = []

    #gets data
    for i in range(0, numRuns):
        print('Running Simulation Number {}'.format(i+1))
        avgs, vars = [], []
        fractions = np.arange(0, 1, 0.05) # 0.05
        for i in range(len(fractions)):
            print('Running Sim for Fraction = {:.2f}'.format(fractions[i]))
            f = fractions[i]
            avgI, varI = withImmune(1000, 100, 1, f)
            avgs.append(avgI)
            vars.append(vars)
        fullAvgs.append(avgs)

    #evaluates errors
    for i in range(len(fullAvgs[0])):
        data  = []
        for j in range(len(fullAvgs)):
            data.append(fullAvgs[j][i])
        errors.append(np.std(data))
        newAvgs.append(np.mean(data))

    np.savetxt('immunitySim.txt', [fractions, avgs, errors])

    plt.errorbar(fractions, newAvgs, yerr = errors)
    plt.ylabel('Avg Infected Sites')
    plt.xlabel('Fraction of Immune Sites')
    plt.title('Immunity Simulation')
    plt.savefig('Immunity Simulation')
    plt.show()


def runVisualisation():
    latticeClass = Lattice(n, 3)
    lat = latticeClass.random

    fig, ax = plt.subplots()
    im = plt.imshow(lat)
    sim = Simulation(lat)
    anim = ani.FuncAnimation(fig, animate, fargs=(im, sim), blit=True, interval = 0.01)
    plt.xlabel("Row")
    plt.ylabel("Column")
    plt.show()


def settings():
    """Takes required parameters from the user and runs the relevant version:

        Takes:
            user input = documented on start up

        Returns:
            None
    """
    global n, p1, p2, p3
    print('\n ###################################################################')
    n = int(input('Please input the size of the system: '))
    print('Would you like to run one of the simulations or a visualisation ?')
    print("Options:\
                    \n    'without' = without immunity simulation\
                    \n    'with' = simulation with immunity\
                    \n    'vis' = simple visualisation with given params\n")
    objective = (input('Selection for Objective: ')).split()[0]
    print('\n ###################################################################')

    if objective == 'without':
        #these values later get redefined
        p1, p2, p3 = 0, 0, 0
        print("Options:\
                        \n    'full' = the full range of p1 and p3 simulated with 1000 sweeps\
                        \n    'zoomed' = the more accurate but smaller range simulation\n")
        simType = (input('Selection for Simulation to Run: ')).split()[0]
        if simType == 'full':
            runFullSim()
        if simType == 'zoomed':
            runZoomedSim()

    if objective == 'with':
        runImmunitySim()

    if objective == 'vis':
        print("""
                                    p1      p2      p3
            Absorbing state:        0.10    0.90    0.90
            Dynamic Equilibrium:    0.50    0.50    0.50
            Waves of Infection:     0.80    0.10    0.01

                """)
        print('Please input p1 p2 and p3, space seperated and as decimals: \n')
        [p1, p2, p3] = (input('Values: ')).split()
        p1, p2, p3 = float(p1), float(p2), float(p3)
        runVisualisation()

settings()

#n = 50
#xVals, yVals, avgs, vars = mainSim([0, 0.5], [0, 0.5], 0.1, 1000, 100, 1)
#plt.contourf(xVals, yVals,  avgs)
#plt.show()

#p1 = 0.5
#p2 = 0.5
#p3 = 0.5
'''
p1 = 0.5
p2 = 0.5
p3 = 0.5
n = 50
latticeClass = Lattice(n, 3)
lat = latticeClass.random

levels = [0, 1, 2, 3]
colours = ['white', 'red', 'gray']
cmap, norm = from_levels_and_colors(levels, colours)

fig, ax = plt.subplots()
im = plt.imshow(lat, cmap = cmap)
sim = Simulation(lat)
anim = ani.FuncAnimation(fig, animate, fargs=(im, sim), interval = 200, blit=True)
plt.xlabel("Column")
plt.ylabel("Row")
plt.title('SIRS Simulation')
plt.show()
'''
