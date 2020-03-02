import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.patches as mpatches
from scipy.ndimage.measurements import center_of_mass

from latticeGenerator import Lattice

'''
Notes:
    Rules:
      - any live cell with less than 2 live neighbours dies
      - any live cell with 2 or 3 live neighbours lives
      - any live cell with more than 3 live neighbours dies
      - any dead cell with exactly 3 neighbours becomes alive

 i = row, j = column
 1 is alive, 0 is dead
 '''

class Simulation:
    def __init__(self, lat):
        self.lat = lat
        self.data = []

    def COM(self):
        return center_of_mass(self.lat)

def nnCoords(lat, i, j):
    """Finds the coordinates of the NNs:

        Takes:
            n(int)(global)= dimension of array, expect shape of lattice(n, n)
            lat(array(n, n)) = the simulation lattice
            i(int) = column index of point
            j(int) = row index of point

        Returns:
            l, r, u, d, lu, ru, ld, rd (size 2 arrays) = The coordinates of the NNs
            siteData (size 8 array) = the values for NNs: l, r, u, d, lu, ru, ld, rd
    """
    l = [i, (j + 1)%n]
    r = [i, (j - 1)%n]
    u = [(i - 1)%n, j]
    d = [(i + 1)%n, j]

    lu = [(i - 1)%n, (j + 1)%n]
    ru = [(i + 1)%n, (j + 1)%n]
    ld = [(i - 1)%n, (j - 1)%n]
    rd = [(i + 1)%n, (j - 1)%n]
    #Stores all of the actual spins in an array for easier use outside function
    siteData = [ lat[(i - 1)%n, j], lat[(i + 1)%n, j], lat[i, (j + 1)%n],
                lat[i, (j - 1)%n], lat[(i - 1)%n, (j + 1)%n],lat[(i + 1)%n, (j + 1)%n],
                lat[(i - 1)%n, (j - 1)%n], lat[(i + 1)%n, (j - 1)%n] ]
    return l, r, u, d, lu, ru, ld, rd, siteData

def totalActive(lat):
    """Finds the total number of active sites in the lattice:

        Takes:
            lat(array(n, n)) = the simulation lattice

        Returns:
            (int) = The total number of active sites
    """
    return np.sum(lat)

def oneSample(lat):
    """Runs one GOL update over the full lattice

        Takes:
            n(int)(global)= dimension of array, expect shape of lattice(n, n)
            lat(array(n, n)) = the simulation lattice

        Returns:
            new(array(n, n)) = the new and updated simulation lattice
    """
    new =  np.random.randint(0, 1, size = n**2)
    new = new.reshape((n, n))
    #iterates over all elements of lattice
    for c in range(0, n):
        for r in range(0, n):
            i, j =  c, r
            #stores whether alive or dead
            AorD = lat[i, j]
            #obtains all of the NN data
            l, r, u, d, lu, ru, ld, rd, siteData = nnCoords(lat, i, j)
            #counts the number of live cells around our current point
            count = sum(siteData)

            #Applies the GOL ruleset
            if AorD == 1:
                if (count == 2) or (count == 3):
                    new[i, j] = 1
                    continue
                else:
                    new[i, j] = 0
            if AorD == 0:
                if (count == 3):
                    new[i, j] = 1
    return new

def runTillEquil(lat, avgSize):
    """Runs a lattice update scheme until 'equlibrium' is met:

        Takes:
            lat(array(n, n)) = the simulation lattice
            avgSize(int) = the size of the list to average to test for
                            equilibrium is

        Returns:
            i(int) = The number of iterations taken to reach equlibrium
    """
    #i stores number of iterations completed
    i = 0
    prevAvg = 0
    while True:
        #averages avgSize number of total active values for comparison
        for j in range(avgSize):
            values = []
            #runs an update
            lat = oneSample(lat)
            #collects the total number of active sites currently in the lattice
            values.append(totalActive(lat))
        #finds the average of the new points sampled
        newAvg = np.mean(values)
        #compares this new avg to the previous one, if same then equilibrium
        if newAvg == prevAvg:
            return i
        #if not the same, then not in equlibrium and set new as old for later comparison
        else:
            i +=1*avgSize
            prevAvg = newAvg
            continue

def timeSim(avgSize, histSize):
    """Runs a lattice update scheme until 'equlibrium' is met:

        Takes:
            avgSize(int) = the size of the list to average to test for
                            equilibrium is
            avgHist(int) = the number of samples to be calculated/ put into
                            histogram

        Outputs:
            graph(1) = a histogram of the number of iterations to equilibrate,
                        figure is also saved.
    """
    print('Starting Data Collection')
    print('0% Complete')
    lengths = []
    #samples histSize number of equilibrium times
    for i in range(1, histSize+1):
        latticeClass = Lattice(n, 2)
        #generates a random lattice
        lat = latticeClass.random
        lengths.append(runTillEquil(lat, avgSize))
        #prints updates at every 10% completed
        if i%10 == 0:
            print('{:.0f}% Complete'.format(float(i/histSize)*100))
    title = 'Equlibration Time for GOL'
    plt.hist(lengths, bins = 20, color = 'b', edgecolor='k', alpha = 0.5)
    plt.xlabel('Iterations to Equilibrate')
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(title)
    plt.show()

def velocity(data):
    """Function calcualtes the average velocity of the spaceship:

        Takes:
            data(array) = a set of centre of mass arrays(size 2)

        Returns:
            average(float) = the average value of the calcualted velocities
    """
    differences = []
    for i in range(len(data)-10):
        value1 = data[i]
        value2 = data[i+10]

        combined = [value1[0], value1[1], value2[0], value2[1]]
        skip = False
        for j in combined:
            if j > 46:
                skip = True
            if j <4:
                skip = True
        if skip == True:
            continue

        diff = [(value2[0] - value1[0])/10, (value2[1] - value1[1])/10]
        if (abs(diff[0]) > 1) or (abs(diff[1]) > 1) or (abs(diff[0]) <0) or (abs(diff[1]) < 0):
            continue
        else:
            differences.append(diff)
    velocities = [np.hypot(*j) for j in differences]
    return np.mean(velocities)


def speedSim(numSteps, incriment, delay = 10):
    """Runs the determination of the spaceship velocity simulation:

        Takes:
            numSteps(int) = the number of differnt values that should be sampled
                            for the resulting plot
            incriment(int) = the spacing in iterations between the steps
            delay(int) = represents the ms delay in displaying a new frame in
                            the animation

        Outputs:
            graph(1) = a line/point plot of the calculated ship velocites vs the
                        number of iterations averaged over, figure is also saved.
    """
    graphTitle = 'Velocity Of Ship Vs Total Iterations'
    limit = 500
    latticeClass = Lattice(n, 2)
    lat = latticeClass.glider

    fig, ax = plt.subplots()
    im = plt.imshow(lat)
    sim = Simulation(lat)
    anim = ani.FuncAnimation(fig, animate, fargs=(im, sim, limit), interval = delay, blit=True)
    plt.ylabel("Row")
    plt.xlabel("Column")
    plt.title('Spaceship Simulation for {} steps'.format(limit))
    plt.show()
    speed = velocity(sim.data)
    print('The Average velocity of the ship is: {:.3f} sites/iteration \n'.format(speed))

def latticeSelect():
    """Allows use to select a system to visualise:

        Takes:
            User input

        Returns:
            lattice((n, n)) = the chosen simulation lattice
    """
    print("options of systems:\
            \n    Still: 'beehive' or 'tub'\
            \n    Oscillating: 'pulsar', 'blinker' or 'penta'\
            \n    Spaceship: 'glider'")
    chosen = input('Please input your choice as a string with no quotations: ')
    latticeClass = Lattice(n, 2)

    if chosen == 'beehive':
        return latticeClass.beehive
    if chosen == 'tub':
        return latticeClass.tub
    if chosen == 'pulsar':
        return latticeClass.pulsar
    if chosen == 'blinker':
        return latticeClass.blinker
    if chosen == 'penta':
        return latticeClass.penta
    if chosen == 'glider':
        return latticeClass.glider

def visualisation(lat, delay = 200):
    """Runs a visualisation for a given lattice and has no set end:

        Takes:
            lat(array(n, n)) = the simulation lattice
            delay(int) = represents the ms delay in displaying a new frame in
                            the animation

        Returns:
            None
    """
    fig, ax = plt.subplots()
    im = plt.imshow(lat)
    sim = Simulation(lat)
    anim = ani.FuncAnimation(fig, animate, fargs=(im, sim, None), interval = delay, blit=True)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title('Game Of Life Simulation')
    plt.show()

def animate(i, im, sim, limit):
    """The animate function that is sent to func animation:

        Takes:
            lat(array(n, n)) = the simulation lattice
            im(matplotlib object) = the plottng screen, an imshow object
            limit(int) = stops the animation when number frames passes this value

        Returns:
            [im] = updated imshow object for animation to use
    """
    #If no limit set we run as normal
    if limit == None:
        new_lat = oneSample(sim.lat)
        sim.lat = new_lat
        sim.data.append(sim.COM())
        im.set_array(new_lat)
        yellowPatch = mpatches.Patch(color='yellow', label='Live Cells')
        purplePatch = mpatches.Patch(color='purple', label='Dead Cells')
        legend = plt.legend(handles=[yellowPatch, purplePatch])
        return [im] + [legend]

    #if limit is passed, the animation shuts down
    if len(sim.data) > limit:
        plt.close()

    new_lat = oneSample(sim.lat)
    sim.lat = new_lat
    #gets the [x, y] centre of mass
    sim.data.append(sim.COM())
    im.set_array(new_lat)
    return [im]

def settings():
    """Takes required parameters from the user and runs the relevant version:

        Takes:
            user input = documented on start up

        Returns:
            None
    """
    global n
    print('\n ###################################################################')
    print('Would you like to run one of the available simulation or a visualisation ?')
    print("Options:\
                    \n    'timeSim' = simulation to find equilibrium times of random systems\
                    \n    'speedSim' = simulation to find the speed of the glider spaceship\
                    \n    'vis' = a visualisation of a chosen system(moves on to next set of options)")
    print('Please input your choice as a single string with no quotation marks. ')
    print('\n ###################################################################')
    info = (input('Selection for Objective: ')).split()

    print('\n ###################################################################')
    n = int(input('Lattice Size (n, n), please input n: '))
    print('\n ###################################################################')
    if len(info) < 1:
        raise Exception("MISSING AT LEAST ONE ARGUMENT, SHOULD BE EXACTLY 1")
        sys.exit()
    if len(info) > 1:
            raise Exception("HAVE BEEN GIVEN AT LEAST ONE EXTRA ARGUMENT, SHOULD BE EXACTLY 1")
            sys.exit()
    objective = info[0]
    if objective == 'vis':
        toBeUsed = latticeSelect()
        visualisation(toBeUsed)
    if objective == 'timeSim':
        #(avgSize, histSize)
        timeSim(10, 100)
    if objective == 'speedSim':
        #(numSteps, incriment, delay = 10)
        speedSim(5, 50, 1)

settings()
