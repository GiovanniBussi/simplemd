import numpy as np
import re
import numba
import time
import tempfile
import sys

# Random number generator compatible with C++ and FORTRAN versions

_IA=16807
_IM=2147483647
_IQ=127773
_IR=2836
_NTAB=32
_NDIV=(1+(_IM-1)//_NTAB)
_EPS=3.0e-16
_AM=1.0/_IM
_RNMX=(1.0-_EPS)

@numba.jit(cache=True,fastmath=True)
def _U01(idum,iy,iv):
    if idum <= 0 or iy == 0:
        if (-idum < 1):
            idum=1
        else:
            idum=-idum
        for j in range(_NTAB+7,-1,-1):
            k=idum//_IQ
            idum=_IA*(idum-k*_IQ)-_IR*k
            if idum<0:
                idum += _IM
            if j < _NTAB:
                iv[j] = idum
        iy=iv[0]
    k=idum//_IQ
    idum=_IA*(idum-k*_IQ)-_IR*k
    if idum<0:
        idum += _IM
    j=iy//_NDIV
    iy=iv[j]
    iv[j]=idum
    temp=_AM*iy
    if temp > _RNMX:
        temp = _RNMX
    return idum,iy,temp

@numba.jit(cache=True,fastmath=True)
def _U01_loop(idum,iy,iv,numbers):
    for i in range(len(numbers)):
        idum,iy,numbers[i] = _U01(idum,iy,iv)
    return idum,iy

@numba.jit(cache=True,fastmath=True)
def _Gaussian(idum,iy,iv,switchGaussian,saveGaussian):
    if(switchGaussian):
        return idum,iy,False,0.0,saveGaussian
    else:
        while True:
            idum,iy,r1=_U01(idum,iy,iv)
            idum,iy,r2=_U01(idum,iy,iv)
            v1=2.0*r1-1.0
            v2=2.0*r2-1.0
            rsq=v1*v1+v2*v2
            if(rsq<1.0 and rsq>0.0):
                break
        fac=np.sqrt(-2.0*np.log(rsq)/rsq)
        return idum,iy,True,v1*fac,v2*fac

@numba.jit(cache=True,fastmath=True)
def _Gaussian_loop(idum,iy,iv,switchGaussian,saveGaussian,numbers):
    for i in range(len(numbers)):
        idum,iy,switchGaussian,saveGaussian,numbers[i] = _Gaussian(
            idum,iy,iv,switchGaussian,saveGaussian
    )
    return idum,iy,switchGaussian,saveGaussian

class Random():
    def __init__(self,seed=0):
        self.switchGaussian=False
        self.saveGaussian=0.0
        self.iy=0
        self.iv=np.zeros(_NTAB, dtype=int)
        self.idum=0
        self.idum=seed
    def U01(self,shape=None):
        if shape is None:
            self.idum,self.iy,temp = _U01(self.idum,self.iy,self.iv)
            return temp
        else:
            numbers=np.zeros(np.prod(shape))
            self.idum,self.iy = _U01_loop(self.idum,self.iy,self.iv,numbers)
            return numbers.reshape(shape)

    def Gaussian(self,shape=None):
        if shape is None:
            self.idum,self.iy,self.switchGaussian,self.saveGaussian,temp=_Gaussian(
                self.idum,self.iy,self.iv,self.switchGaussian,self.saveGaussian
            )
            return temp
        else:
            numbers=np.zeros(np.prod(shape))
            self.idum,self.iy,self.switchGaussian,self.saveGaussian=_Gaussian_loop(
                self.idum,self.iy,self.iv,self.switchGaussian,self.saveGaussian,numbers
            )
            return numbers.reshape(shape)


@numba.jit(cache=True,fastmath=True)
def _compute_forces(cell, positions, forcecutoff, neighbors, point, forces):
    engconf=0.0
    forces.fill(0.0)
    forcecutoff2=forcecutoff*forcecutoff
    engcorrection=4.0*(1.0/forcecutoff2**6-1.0/forcecutoff2**3)

    for i in range(len(positions)):
        for j in range(point[i],point[i+1]):
            ja=neighbors[j]
            distancex=positions[i,0]-positions[ja,0]
            distancey=positions[i,1]-positions[ja,1]
            distancez=positions[i,2]-positions[ja,2]
            distancex-=np.floor(distancex/cell[0]+0.5)*cell[0]
            distancey-=np.floor(distancey/cell[0]+0.5)*cell[1]
            distancez-=np.floor(distancez/cell[0]+0.5)*cell[2]
            distance2=distancex**2+distancey**2+distancez**2
            if distance2 <= forcecutoff2:
                invdistance2=1.0/distance2
                invdistance6=invdistance2*invdistance2*invdistance2
                e=4.0*invdistance6*invdistance6-4.0*invdistance6-engcorrection
                engconf+=e
                fmod=2.0*4.0*(6.0*invdistance6*invdistance6-3.0*invdistance6)*invdistance2
                fx=fmod*distancex
                fy=fmod*distancey
                fz=fmod*distancez
                forces[i,0]+=fx
                forces[i,1]+=fy
                forces[i,2]+=fz
                forces[ja,0]-=fx
                forces[ja,1]-=fy
                forces[ja,2]-=fz
    return engconf

@numba.jit(cache=True,fastmath=True)
def _compute_list(cell,positions,listcutoff,nlist,point):
   listcutoff2=listcutoff**2
   point[0]=0
   for i in range(len(positions)):
       point[i+1]=point[i]
       for j in range(i+1,len(positions)):
            distancex=positions[i,0]-positions[j,0]
            distancey=positions[i,1]-positions[j,1]
            distancez=positions[i,2]-positions[j,2]
            distancex-=np.floor(distancex/cell[0]+0.5)*cell[0]
            distancey-=np.floor(distancey/cell[0]+0.5)*cell[1]
            distancez-=np.floor(distancez/cell[0]+0.5)*cell[2]
            distance2=distancex**2+distancey**2+distancez**2
            if distance2 <= listcutoff2:
                if point[i+1]>=len(nlist):
                    raise Exception("Verlet list size exceeded\nIncrease maxneighbours")
                nlist[point[i+1]]=j
                point[i+1]+=1


class SimpleMD:
    def __init__(self):
        self.iv=np.zeros(32,dtype=int)
        self.iy=0
        self.iset=0
        self.gset=0.0
        self.write_positions_first=True
        self.write_statistics_first=True
        self.write_statistics_last_time_reopened=0
        self.write_statistics_fp=None

        self.temperature=1.0
        self.maxneighbors=1000
        self.tstep=0.005
        self.friction=0.0
        self.forcecutoff=2.5
        self.listcutoff=3.0
        self.nstep=1
        self.nconfig=10
        self.nstat=1
        self.idum=0
        self.wrapatoms=False
        self.statfile=""
        self.trajfile=""
        self.outputfile=""
        self.inputfile=""

        self.statfile_f=None


    def read_input(self,file):
        with open(file,"r") as f:
            for line in f:
                line=re.sub("#.*$","",line)
                line=re.sub(" *$","",line)
                words=line.split()
                if len(words)==0:
                    continue
                key=words[0]
                if key=="temperature":
                    self.temperature=float(words[1])
                elif key=="tstep":
                    self.tstep=float(words[1])
                elif key=="friction":
                    self.friction=float(words[1])
                elif key=="forcecutoff":
                    self.forcecutoff=float(words[1])
                elif key=="listcutoff":
                    self.listcutoff=float(words[1])
                elif key=="nstep":
                    self.nstep=int(words[1])
                elif key=="nconfig":
                    self.nconfig=int(words[1])
                    self.trajfile=words[2]
                elif key=="nstat":
                    self.nstat=int(words[1])
                    self.statfile=words[2]
                elif key=="wrapatoms":
                    if re.match("[Tt].*",words[1]):
                        self.wrapatoms=True
                elif key=="maxneighbours":
                    self.maxneighbors=int(words[1])
                elif key=="inputfile":
                    self.inputfile=words[1]
                elif key=="outputfile":
                    self.outputfile=words[1]
                elif key=="idum":
                    self.idum=int(words[1])
                else:
                    raise Exception("Unknown keyword: "+key)
        if len(self.inputfile)==0:
            raise Exception("Specify input file")
        if len(self.outputfile)==0:
            raise Exception("Specify output file")
        if len(self.trajfile)==0:
            raise Exception("Specify traj file")
        if len(self.statfile)==0:
            raise Exception("Specify stat file")

    def read_positions(self,file):
        with open(file,"r") as f:
            natoms=int(f.readline())
            cell=[float(x) for x in f.readline().split()]
            positions=np.loadtxt(f,usecols=(1,2,3))
        assert(len(positions)==natoms)
        return np.array(cell),np.array(positions)

    def randomize_velocities(self,temperature,masses,random):
       return np.sqrt(temperature/masses)[:,np.newaxis]*random.Gaussian(shape=(len(masses),3))

    # note: this can act on a vector of vectors
    def pbc(self,cell,vector):
        return vector-np.floor(vector/cell+0.5)*cell

    def check_list(self,positions,positions0,listcutoff,forcecutoff):
       delta2=(0.5*(listcutoff-forcecutoff))*(0.5*(listcutoff-forcecutoff))
       disp2=np.sum((positions-positions0)**2,axis=1)
       return np.any(disp2>delta2)

    def compute_engkin(self,masses,velocities):
        return 0.5*np.sum(masses*np.sum(velocities**2,axis=1))

    def thermostat(self,masses,dt,friction,temperature,velocities,engint,random):
        c1=np.exp(-friction*dt)
        c2=np.sqrt((1.0-c1*c1)*temperature)/np.sqrt(masses)
        engint+=0.5*np.sum(masses*np.sum(velocities**2,axis=1))
        velocities=c1*velocities+c2[:,np.newaxis]*random.Gaussian(shape=velocities.shape)
        engint-=0.5*np.sum(masses*np.sum(velocities**2,axis=1))
        return velocities,engint

    def write_positions(self,cell,positions,wrapatoms=False):
        mode="w"
        if self.write_positions_first:
            self.write_positions_first = False
            mode="w"
        else:
            mode="a"

        with open(self.trajfile,mode) as f:
            print("%d" % len(positions), file=f)
            print("%f %f %f" % (cell[0], cell[1], cell[2]), file=f)
            if wrapatoms:
                positions = pbc(cell,positions)
            np.savetxt(f,positions,fmt="Ar %10.7f %10.7f %10.7f")

    def write_final_positions(self,cell,positions,wrapatoms=False):
        with open(self.outputfile,"w") as f:
            print("%d" % len(positions), file=f)
            print("%f %f %f" % (cell[0], cell[1], cell[2]), file=f)
            if wrapatoms:
                positions = pbc(cell,positions)
            np.savetxt(f,positions,fmt="Ar %10.7f %10.7f %10.7f")

    def write_statistics(self,istep,tstep,natoms,engkin,engconf,engint):
         if self.write_statistics_fp is None:
             self.write_statistics_fp = open(self.statfile, "w")
         if istep-self.write_statistics_last_time_reopened>100:
             self.write_statistics_fp.close()
             self.write_statistics_fp = open(self.statfile, "a")
             self.write_statistics_last_time_reopened=istep
         print("%d %f %f %f %f %f" %
               (istep,istep*tstep,2.0*engkin/(3.0*natoms),engconf,engkin+engconf,engkin+engconf+engint),
               file=self.write_statistics_fp)

    def run(self,parameters):
        self.read_input(parameters)
        cell,positions=self.read_positions(self.inputfile)
        random=Random(self.idum)

        # masses are hardcoded to 1
        masses=np.ones(len(positions))

        # energy integral initialized to 0
        engint=0.0

        # velocities are randomized according to temperature
        velocities=self.randomize_velocities(self.temperature,masses,random)

        # allocate space for neighbor lists
        nlist=np.zeros(self.maxneighbors*len(positions), dtype=int)
        point=np.zeros(len(positions)+1, dtype=int)
        # neighbour list are computed
        _compute_list(cell, positions, self.listcutoff, nlist, point)

        print("Neighbour list recomputed at step ",0)
        print("List size: ",len(nlist))

        # reference positions are saved
        positions0=+positions

        forces=np.zeros(shape=positions.shape)

        # forces are computed before starting md
        engconf= _compute_forces(cell, positions, self.forcecutoff, nlist, point, forces)

        # here is the main md loop
        # Langevin thermostat is applied before and after a velocity-Verlet integrator
        # the overall structure is:
        #   thermostat
        #   update velocities
        #   update positions
        #   (eventually recompute neighbour list)
        #   compute forces
        #   update velocities
        #   thermostat
        #   (eventually dump output informations)

        now=time.time()
        for istep in range(self.nstep):
            if self.friction>0:
                velocities,engint = self.thermostat(
                        masses,0.5*self.tstep,self.friction,self.temperature,velocities,engint,random)

            velocities+=forces*0.5*self.tstep/masses[:,np.newaxis]
            positions+=velocities*self.tstep

            check_list=self.check_list(positions,positions0,self.listcutoff,self.forcecutoff)
            if check_list:
                _compute_list(cell, positions, self.listcutoff, nlist, point)
                positions0=+positions
                print("Neighbour list recomputed at step ",istep)
                print("List size: ",len(nlist))

            engconf = _compute_forces(cell, positions, self.forcecutoff, nlist, point, forces)

            velocities+=forces*0.5*self.tstep/masses[:,np.newaxis]

            if self.friction>0.0:
                velocities,engint = self.thermostat(
                        masses,0.5*self.tstep,self.friction,self.temperature,velocities,engint,random)

            if (istep+1)%self.nconfig==0:
                self.write_positions(cell,positions,self.wrapatoms)
            if (istep+1)%self.nstat==0:
                engkin = self.compute_engkin(masses,velocities)
                self.write_statistics(istep+1,self.tstep,len(positions),engkin,engconf,engint)

        self.write_final_positions(cell,positions,self.wrapatoms)

        if self.write_statistics_fp is not None:
            self.write_statistics_fp.close()

        print(time.time()-now)

if __name__ == "__main__":
    # read from stdin and store on a temporary file
    input = sys.stdin.read()
    with tempfile.NamedTemporaryFile("w+t") as tmp:
        tmp.write(input)
        tmp.flush()
        simplemd=SimpleMD()
        simplemd.run(tmp.name)

