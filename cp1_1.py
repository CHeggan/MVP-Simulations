import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani

#Object class, creates and stores a variety of lattice configs upon creation
class Lattice:
    def __init__(self, n):
        self.n = n
        self.randomLattice = self.createRandom()
        self.allUp = np.ones((n, n))
        self.allDown = self.makeSpins(np.zeros((n**2, 1)))
        self.halfUp = self.createHalf()

    def createRandom(self):
        """Creates a random initial lattice with -1 and 1 spins"""
        lattice = np.random.uniform(0, 1, self.n**2)
        lattice = self.makeSpins(lattice)
        return lattice

    def createHalf(self):
        size =int(self.n**2/2)
        firstPart = np.random.randint(-1, -0, size)
        secondPart = np.random.randint(1, 2, size)

        newArray = np.concatenate((firstPart, secondPart))
        newArray = newArray.reshape((n, n))
        return newArray


    def makeSpins(self, lattice):
        """The function that is called by 'create_random' to actually set spins"""
        #sets the spin = 1 if random num bigger than 0.5
        for i in range(len(lattice)):
            if lattice[i] > 0.5:
                lattice[i] = 1
            #otherwise set as -1
            else:
                lattice[i] = -1
        #makes the lattice a (50,50) as opposed to (2500, 1)
        lattice = lattice.reshape((self.n, self.n))
        return lattice

class Simulation:
    def __init__(self, lat):
        self.lat = lat

################################################################################
def nnCoords(lat, i, j):
    """Finds the coordinates of the NNs:

        Takes:
            n(int)(global)= dimension of array, expect shape of lattice(n, n)
            lat(array(n, n)) = the simulation lattice
            i(int) = column index of point
            j(int) = row index of point

        Returns:
            l, r, u, d (size 2 arrays) = The coordinates of the NNs
            spins (size 4 array) = the spins at the NN coords, goes l, r, u, d
    """
    l = [(i - 1)%n, j]
    r = [(i + 1)%n, j]
    u = [i, (j + 1)%n]
    d = [i, (j - 1)%n]

    #Stores all of the actual spins in an array for easier use outside function
    spins = [lat[(i - 1)%n, j], lat[(i + 1)%n, j], lat[i, (j + 1)%n], lat[i, (j - 1)%n]]
    return l, r, u, d, spins

def totalEnergy(lat):
    """Finds the total energy of a lattice configuration:

        Takes:
            lat(array(n, n)) = the simulation lattice

        Returns:
            total(int) = the total energy of the lattice provided
    """
    total = 0
    #iterates over every element of the array, first going down then across
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            #finds all of the NNs
            l, r, u, d, spins = nnCoords(lat, i, j)
            #We only sum over the left and up across the entire array to avoid double counting
            #calculates the energy using equation E = -J(sum(si*sj))
            E = -1 * lat[i, j] * (spins[0] + spins[2])
            total += E
    return total

def totalMag(lat):
        """Finds the absolute total magnetisation of a lattice configuration:

            Takes:
                lat(array(n, n)) = the simulation lattice

            Returns:
                (int) = the total absolute magnetisation of the lattice provided
        """
        return np.abs(np.sum(lat))/n**2

def acceptOrNah(delE):
    """Decides whether we accept the flip or not:

        Takes:
            delE(float) = the energy difference of the changed flip(s)

            Returns:
                (boolean) = True(accept) or false(don't accept)

    """
    #alwats accept if the energy goes down
    if delE <=0:
        return True
    #if energy goes up we draw a random number and compare to the probability
    prob = np.exp(-delE/T)
    randNum = np.random.uniform()
    #If random number is smaller than the prob, we accept the flip
    if prob > randNum:
        return True
    else:
        return False

def calcDelE(lat, i, j):
    """Calculated the energy difference form the old state to the new:

        Takes:
            lat(array(n, n)) = the simulation lattice

        Returns:
            delE(int) = the calculated energy difference
            l, r, u, d (size 2 arrays) = The coordinates of the NNs
    """
    l, r, u, d, spins = nnCoords(lat, i, j)
    #energy equation: DelE = -2J(Si)[s(nn1) + s(nn2) + s(nn3) + s(nn4)], J = 1
    delE = -2*lat[i, j] * (sum(spins))
    #returns the nn coords as well for comparision is kawasaki dynamics
    return delE, l, r, u, d, spins

def sweep(lat, function):
    """Finds the coordinates of the NNs:

        Takes:
            n(int)(global)= dimension of array, expect shape of lattice(n, n)
            lat(array(n, n)) = the simulation lattice
            function(function) = the dynamics function to be used

        Returns:
            lat(array(n, n)) = the updated simulation lattice
    """
    for i in range(n**2):
        lat = function(lat)
    return lat

def perTemp(lat, dynamics, totSweeps, equil, spacing):
    """Calculates relevant data for a particular dynamical system at temp T:

        Takes:
            T(float)(global) = temperate at which the system is operating
            lat(array(n, n)) = the simulation lattice
            dynamics(function) = dynamics function to be used
            totSweeps(int) = total number of sweeps each temp should run for
            equil(int) = number of sweeps that should be used for equilibrisation
            spacing(int) = number of sweeps in between measurements, there are
                        (totSweeps - equil)/spacing measurements

        Returns:
            avgM(float) = average absolute os Magnetisations
            avgE(float) = average of total energies
            chi(float) = magnetic suseptibily
            cap(float) = heat capacity
    """
    eVals = []
    mVals = []
    #runs equil sweeps to set up system properly before sampling results
    for i in range(equil):
        lat = sweep(lat, dynamics)
    #Runs 'totSweepsl' more sweeps where we sample every 'spacing' sweeps
    for j in range(totSweeps):
        lat = sweep(lat, dynamics)
        if j%10 == 0:
            mag = totalMag(lat)
            ene = totalEnergy(lat)
            eVals.append(ene)
            mVals.append(mag)
            print('Temp: {:.1f}, Sweep: {}, Mag: {}, Ene: {}'.format(T, j, mag, ene))
    #calculates the M and E averages
    avgE = np.mean(eVals)
    avgM = np.mean(mVals)
    #calculates suseptibily and heat capacity making use of numpy's variance
    cap = np.var(eVals)/(T**2 * len(eVals))
    chi = np.var(mVals)/(T*len(mVals))

    #Gets the errors using the jacknife method
    eErr, mErr, capError, chiError = jacknife(mVals, eVals)
    errors = {'eErr':eErr, 'mErr':mErr, 'capError':capError, 'chiError':chiError}

    return avgM, avgE, chi, cap, lat, errors

def jacknife(mVals, eVals):
    """Calculates the errors on each temperature point using the jacknife sampling
        method. Works by iteratively removing points from the results arrays and
        recalculating heat capacity and magnetic suseptibility:

        Takes:
            mVals(array) = array of all sampled total magenetisms of the system
            eVals(array) = array of all sampled total energies of the system

        Returns:
            eErr(float) = the statistical error on the average energy point
            mErr(float) = ststistical error on the average magnetisation
            capError(float) = ststistical error on the heat capacity
            chiError(float) = ststistical error on the megnetic suseptibility
    """
    #For both <E> and <M>, standard deviation is reasonable
    eErr = math.sqrt( np.var(eVals)/ len(eVals))
    mErr = math.sqrt( np.var(mVals)/ len(mVals))

    capList = []
    chiList = []
    #Go over the lists and remove ith element at a time, so new array is always size len - 1
    for i in range(len(eVals)):
        #remove ith element and recalculate heat capcity and Susceptibility
        newEvals = np.delete(eVals, i)
        newMvals = np.delete(mVals, i)

        #New calculalted cap and chi added to the lists created earlier
        nextCap = np.var(newEvals)/(T**2 * len(newEvals))
        nextChi = np.var(newMvals)/(T*2 * len(newMvals))

        #new lists should end up same length as eVals and mVals
        capList.append(nextCap)
        chiList.append(nextChi)

    #The errors on heat capcity and chi are sqrt(size of array)*standard deviation without sqrt(1/m) factor
    capError = math.sqrt(len(eVals))*math.sqrt(np.var(capList))
    chiError = math.sqrt(len(mVals))*math.sqrt(np.var(chiList))
    return eErr, mErr, capError, chiError

def glauber(lat):
    """Glauber dynamics function, attempts one spin flip per run:

        Takes:
            lat(array(n, n)) = the simulation lattice

        Returns:
            lat(array(n, n)) = updated lattice after one attempted flip
    """
    #gets two random number in an array within confines of lattice size
    indices =  np.random.randint(0, high=n, size=2)
    x, y =  indices[0], indices[1]
    #stores the original spin of the lattice site
    OGValue = lat[x, y]
    #changes over spins to the other
    if lat[x, y] == 1:
        lat[x, y] = -1
    else:
        lat[x, y] = 1

    #gets the change in energy and the full nn coord set
    delE, l, r, u, d, spins = calcDelE(lat, x, y)
    #decides whether we accept the flipped spin
    result = acceptOrNah(delE)
    #if we dont accept, default back to original value
    if result == False:
        lat[x, y] = OGValue
    return lat

def kawasaki(lat):
    """Kawasaki dynamics function, attempts one spin swap per run:

        Takes:
            lat(array(n, n)) = the simulation lattice

        Returns:
            lat(array(n, n)) = updated lattice after one attempted spin swap
    """
    #generates two random coordinates for two lattice sites
    index_1 = np.random.randint(0, high = n, size = 2)
    index_2 = np.random.randint(0, high= n, size=2)

    #unpacks these coords for use in out specific functions later on
    x1, y1, x2, y2 = index_1[0], index_1[1], index_2[0], index_2[1]
    #stores the values of our two sites to begin with
    OG1 = lat[x1, y1]
    OG2 = lat[x2, y2]

    #if the spins are the same, changing them has no effect
    if OG1 == OG2:
        return lat

    #Cant start from a all up or all down due to this statement
    #swaps the spin values if different
    lat[x1, y1], lat[x2, y2] = OG2, OG1

    #calculates a change of energy for both spin changes
    delE1, l, r, u, d, spins = calcDelE(lat, x1, y1)
    delE2, l2, r2, u2, d2, spins2 = calcDelE(lat, x2, y2)
    nn1_set = [l, r, u, d]
    nn2_set = [l2, r2, u2, d2]
    #adds the change in energies
    totalDelE = sum(spins)*OG1 + sum(spins2)*OG2 - sum(spins2)*OG1 - sum(spins)*OG2
    #totalDelE = delE1 + delE2

    nn = False
    #if the two points are nn using the coords, set nn to true
    for i in nn1_set:
        for j in nn2_set:
            if i == j:
                nn = True
    #if nn, take away -J(spin1*spin2) where spin1 and 2 are OG values
    if nn == True:
        doubleCount = -1*(OG1 * OG2)
        totalDelE = totalDelE - doubleCount
    #send the total energy change to acceptOrNah function
    result = acceptOrNah(totalDelE)

    #if we dont accept, we set each site back to its starting value
    if result == False:
        lat[x1, y1], lat[x2, y2] = OG1, OG2
    #otherwise send back the new lattice
    return lat

def tempSim(limits, interval, dynamics, totSweeps, equil, spacing):
    """Runs the full temperature simulation:

        Takes:
            limits(size 2 array) = low(index 0) and high(index 1) temp limits
            interval(float or int) = the space between sampled temp values
            T(float)(global) = temperate at which the system is operating
            dynamics(function) = dynamics function to be used
            totSweeps(int) = total number of sweeps each temp should run for
            equil(int) = number of sweeps that should be used for equilibrisation
            spacing(int) = number of sweeps in between measurements, there are
                        (totSweeps - equil)/spacing measurements

        Returns:
            temps(array) = created np.arange array with set limits and spacing
            avgMs(array) = correspnding average magnetisation values to T values
            avgEs(array) = correspnding average energy values to T values
            chis(array) = correspnding magnetic suseptibility values to T values
            capss(array) = correspnding heat capacity values to T values
    """
    global T
    temps = np.arange(limits[0], limits[1], interval)
    avgMs, avgEs, chis, caps  = [], [], [], []
    mErrs, eErrs, chiErrs, capErrs = [], [], [], []

    latticeClass = Lattice(n)
    #if dynamics is galuber we start from an already all up or all down system
    if dynamics == glauber:
        lat = latticeClass.allUp
    #if kawasaki we need to start from half up half down
    else:
        lat = latticeClass.halfUp

    for i in temps:
        T = i
        print('Starting the T = {:.1f} Simulation'.format(T))
        avgM, avgE, chi, cap, lat, perErrors = perTemp(lat, dynamics, totSweeps, equil, spacing)
        avgMs.append(avgM), avgEs.append(avgE), chis.append(chi), caps.append(cap)
        eErrs.append(perErrors['eErr']), mErrs.append(perErrors['mErr'])
        chiErrs.append(perErrors['chiError']), capErrs.append(perErrors['capError'])
    errors = [mErrs, eErrs, chiErrs, capErrs]

    column_names = ['Temp', 'Avg E', 'E Err', 'Avg M', 'M Err', 'Chi', 'Chi Err', 'Cap', 'Cap Err']
    df = pd.DataFrame(columns = column_names)
    df['Temp'] = temps
    df['Avg E'], df['Avg M'], df['Chi'], df['Cap'] = avgEs, avgMs, chis, caps
    df['E Err'], df['M Err'], df['Chi Err'], df['Cap Err'] = eErrs, mErrs, chiErrs, capErrs
    fileName = "rawData" + ".csv"
    df.to_csv(path_or_buf = fileName, index = False)
    return temps, avgMs, avgEs, chis, caps, errors

def animate(i, function, im, sim):
    """The animate function that is sent to func animation:

        Takes:
            lat(array(n, n)) = the simulation lattice
            function(Function) = the dynamics function to be used
            im(matplotlib object) = the plottng screen, an imshow object

        Returns:
            lat(array(n, n)) = the updated simulation lattice
    """
    new_lat = sweep(sim.lat, function)
    sim.lat = new_lat
    im.set_array(new_lat)
    return [im]

def plotNsave(temps, avgMs, avgEs, chis, caps, errors, dynamics):
    """
    Plots the physical data calulated form the simulation along with error bars.

    Takes:
        temps(array) = linspaced array, has all simulated temperatures
        avgMs(array) = average magnetisatio values
        avgEs(array) = average energy values
        chis(array) = magnetic suseptibility values
        caps(array) = heat capacity values
        errors(len 4 array) = ecah element has a full array in place with error values
                for each of our 4 physical values.
        dynamics(function) = the type of dynamics to use to run the simulation

    Outputs:
        graphs(2 or 4) = graphs of important physical values as function of temp,
            2 graphs saved for kawasaki and 4 for glauber
        Text files(2 or 4) = the data plotted in the output graphs
    """
    if dynamics == glauber:
        title1 = 'Average E vs Temp - Glauber'
        plt.errorbar(temps, avgEs, xerr = None, yerr = errors[1], ecolor = 'red', capthick = 1)
        plt.title(title1)
        plt.xlabel('Temperature')
        plt.ylabel('Average Energy of System')
        plt.savefig(title1)
        np.savetxt(title1  + ".txt", [temps, avgEs, errors[1]])
        plt.show()

        title2 = 'Average M vs Temp - Glauber'
        plt.errorbar(temps, avgMs, xerr = None, yerr = errors[0], ecolor = 'red', capthick = 1)
        plt.title(title2)
        plt.xlabel('Temperature')
        plt.ylabel('Average Magnetisation of System')
        plt.savefig(title2)
        np.savetxt(title2   + ".txt", [temps, avgMs, errors[0]])
        plt.show()

        title3 = 'Magnetic Susceptibility vs Temperate - Glauber'
        plt.title(title3)
        plt.xlabel('Temperature')
        plt.ylabel('Magnetic Susceptibility of System')
        plt.errorbar(temps, chis, xerr = None, yerr = errors[2], ecolor = 'red', capthick = 1)
        plt.savefig(title3)
        np.savetxt(title3  + ".txt", [temps, chis, errors[2]])
        plt.show()

        title4 = 'Heat Capacity vs Temperature - Glauber'
        plt.title(title4)
        plt.xlabel('Temperature')
        plt.ylabel('Heat Capacity of System')
        plt.errorbar(temps, caps, xerr = None, yerr = errors[3], ecolor = 'red', capthick = 1)
        np.savetxt(title4  + ".txt" , [temps, caps, errors[3]])
        plt.savefig('caps')
        plt.show()

    else:
        title1 = 'Average E vs Temp - Kawasaki'
        plt.errorbar(temps, avgEs, xerr = None, yerr = errors[1], ecolor = 'red', capthick = 1 )
        plt.title(title1)
        plt.xlabel('Temperature')
        plt.ylabel('Average Energy of System')
        plt.savefig(title1)
        plt.show()

        title2 = 'Heat Capacity vs Temperature - Kawasaki'
        plt.title(title2)
        plt.xlabel('Temperature')
        plt.ylabel('Heat Capacity of System')
        plt.errorbar(temps, caps, xerr = None, yerr = errors[3], ecolor = 'red', capthick = 1 )
        plt.savefig(title2)
        plt.show()

def simulation(size, dynamics):
    """
    Runs the simulation part of this code, no visual is given but command line
    printouts are.

    Takes:
        n(int) = the size of the simulation array(n, n)
        dynamics(function) = the type of dynamics to use to run the simulation

    Returns:
        None

    File Creation:
        data text files(2 or 4) = the data used in the output graphs
    """
    global n
    n = size
    temps, avgMs, avgEs, chis, caps, errors = tempSim([1, 3.1], 0.1, dynamics, 10000, 100, 10)
    plotNsave(temps, avgMs, avgEs, chis, caps, errors, dynamics)

def animation(size, temp, dynamics):
    global n, T
    n = size
    T = temp
    latticeClass = Lattice(n)

    #if glauber start form all one state
    if dynamics == glauber:
        title = 'Glauber'
        lat = latticeClass.randomLattice
    #if kawasaki we need to start from half up half down
    else:
        title = 'Kawasaki'
        lat = latticeClass.halfUp

    fig, ax = plt.subplots()
    im = plt.imshow(lat)
    sim = Simulation(lat)
    anim = ani.FuncAnimation(fig, animate, fargs=(glauber, im, sim), blit=True)
    plt.title("Monte Carlo Ising Model Animation ({} Dynamics)".format(title))
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
    print('\n ###################################################################')
    print("Input Arguments are as follows: \n    n(int) = size of lattice(n, n) \
            \n    T(float or int) = temperate of the system \
             \n    dynamics('g' or 'k') = sets the type of dynamics\
             \n    objective('vis' or 'sim') = choose between an animation or full simulation \
             \n    NOTE: If 'sim' is chosen T is irrelevant\n")
    print("Please input them in the form:\
        \n    'n T dynamics choice'")
    print('\n ###################################################################')

    info = (input('Inputs: ')).split()
    if len(info) < 4:
        raise Exception("MISSING AT LEAST ONE ARGUMENT, SHOULD BE EXACTLY 4")
        sys.exit()
    if len(info) > 4:
            raise Exception("HAVE BEEN GIVEN AT LEAST ONE EXTRA ARGUMENT, SHOULD BE EXACTLY 4")
            sys.exit()

    n = int(info[0])
    T = float(info[1])

    if(n**2%2 != 0):
        raise Exception("N**2 MUST BE DIVISIBLE BY 2")
    if info[2] == 'g':
        dynamics = glauber
    else:
        dynamics = kawasaki

    if info[3] == 'vis':
        animation(n, T, dynamics)
    else:
        simulation(n, dynamics)

settings()
