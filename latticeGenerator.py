import numpy as np

class Lattice:
    """Class takes the total size of the array and the number of types to return
        either a ranodm lattice generation or one of a selected few pre-genned types
        spawned at a ranodm location within the lattice:

        Takes:
            n(int) = size of the array(n, n)
            species(int > 0) = the number of different types of values that can
                                be in the simulation lattice

        Stored:
            A selection of random and hand made/pseudo random lattice formations
    """
    def __init__(self, n, species):
         self.n = n
         self.species = species
         self.random =  self.latticeGen(n)
         self.beehive = self.createBeehive(n)
         self.tub = self.createTub(n)
         self.blinker = self.createBlinker(n)
         self.pulsar = self.createPulsar(n)
         self.penta = self.createPenta(n)
         self.glider = self.createGlider(n)

    def createBlank(self):
        """Generates a lattice(n, n) with only zeros contained:

            Takes:
                n(int)(global) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the blank simulation lattice
        """
        lattice =  np.random.randint(0, 1, size = self.n**2)
        lattice = lattice.reshape((self.n, self.n))
        return lattice

    def latticeGen(self, n):
        """Generates a lattice(n, n) with int(species) types of values contained:

            Takes:
                n(int) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the generated random lattice
        """
        lattice =  np.random.randint(0, self.species, size = self.n**2)
        lattice = lattice.reshape((self.n, self.n))
        return lattice

    def createBlinker(self, n):
        """Generates an oscillating blinker for GOL sim:
            Period = 1

            Takes:
                n(int) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the simulation lattice
        """
        base = self.createBlank()
        ranNum = np.random.randint(0, high = self.n, size = 2)
        i, j= ranNum[0], ranNum[1]

        upDown = np.random.randint(0, high = 2)
        #Makes it so blinker can start horizontal or vertical
        f = [upDown, 1-upDown]
        for k in range(1, 4):
            base[(i + f[0]*k)%n, (j + f[1]*k)%n] = 1
        return base

    def createPulsar(self, n):
        """Generates an oscillating pulsar for GOL sim:
            Period = 3

            Takes:
                n(int) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the simulation lattice
        """
        base = self.createBlank()
        ranNum = np.random.randint(0, high = self.n, size = 2)
        i, j= ranNum[0], ranNum[1]

        #Starting in the middle of the pattern, can mirror in all sectors [i, j]
        combos = [ [1, 1], [1, -1], [-1, 1], [-1, -1] ]
        #just a shorthand way of manually retyping a selection of points
        for m in combos:
            fi, fj = m[0], m[1]
            for k in [1, 6]:
                base[(i + fi*2)%n, (j + fj*k)%n] = 1
                base[(i + fi*3)%n, (j + fj*k)%n] = 1
                base[(i + fi*4)%n, (j + fj*k)%n] = 1

                base[(i + fi*k)%n, (j + fj*2)%n] = 1
                base[(i + fi*k)%n, (j + fj*3)%n] = 1
                base[(i + fi*k)%n, (j + fj*4)%n] = 1
        return base

    def createPenta(self, n):
        """Generates an oscillating penta-decathlon for GOL sim:
            Period = 15

            Takes:
                n(int) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the simulation lattice
        """
        base = self.createBlank()
        ranNum = np.random.randint(0, high = self.n, size = 2)
        i, j= ranNum[0], ranNum[1]

        upDown = np.random.randint(0, high = 2)
        #can spawn either horizontal or vertical
        f = [upDown, 1-upDown]
        #Creates a line 10 long which decays into penta-decathlon
        for k in range(1, 11):
            base[(i + f[0]*k)%n, (j + f[1]*k)%n] = 1
        return base

    def createGlider(self, n):
        """Generates a glider spaceship for GOL sim:

            Takes:
                n(int) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the simulation lattice
        """
        base = self.createBlank()
        ranNum = np.random.randint(0, high = self.n, size = 2)
        i, j= ranNum[0], ranNum[1]
        #manually sets all re-typings, cant find a pattern to make easier
        base[(i)%n, (j+1)%n] = 1
        base[(i)%n, (j-1)%n] = 1
        base[(i+1)%n, (j)%n] = 1
        base[(i + 1)%n, (j + 1)%n] = 1
        base[(i -1)%n, (j + 1)%n] = 1
        return base

    def createBeehive(self, n):
        """Generates a still beehive for GOL sim:

            Takes:
                n(int) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the simulation lattice
        """
        base = self.createBlank()
        ranNum = np.random.randint(0, high = self.n, size = 2)
        i, j= ranNum[0], ranNum[1]
        #manually sets the required points
        base[(i + 1)%n, (j)%n] = 1
        base[(i - 1)%n, (j)%n] = 1
        base[(i)%n, (j - 1)%n] = 1
        base[(i)%n, (j + 2)%n] = 1
        base[(i + 1)%n, (j + 1)%n] = 1
        base[(i -1)%n, (j + 1)%n] = 1
        return base

    def createTub(self, n):
        """Generates a still tub for GOL sim:

            Takes:
                n(int) = size of the array(n, n)

            Returns:
                lattice(array(n, n)) = the simulation lattice
        """
        base = self.createBlank()
        ranNum = np.random.randint(0, high = self.n, size = 2)
        i, j= ranNum[0], ranNum[1]
        #is a simple plus around the central point
        base[(i + 1)%n, (j)%n] = 1
        base[(i - 1)%n, (j)%n] = 1
        base[(i)%n, (j + 1)%n] = 1
        base[(i)%n, (j - 1)%n] = 1
        return base
