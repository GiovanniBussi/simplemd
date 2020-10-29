import sys
import numpy as np

def generate_lattice(n=1, a0=1.6796):
    coord=np.zeros((4*n*n*n,3))
    l=0
    for i in range(n):
       for j in range(n):
           for k in range(n):
               coord[l]=(i,j,k)
               l+=1
               coord[l]=(i+0.5,j,k+0.5)
               l+=1
               coord[l]=(i+0.5,j+0.5,k)
               l+=1
               coord[l]=(i,j+0.5,k+0.5)
               l+=1
    return ((n*a0,n*a0,n*a0),coord*a0)

if __name__ == "__main__":
    if len(sys.argv)<2:
        coord=generate_lattice()
    elif len(sys.argv)<3:
        coord=generate_lattice(int(sys.argv[1]))
    else:
        coord=generate_lattice(int(sys.argv[1]),float(sys.argv[2]))
    print(len(coord[1]))
    print(coord[0][0],coord[0][1],coord[0][2])
    for i in range(len(coord[1])):
        print("Ar",coord[1][i,0],coord[1][i,1],coord[1][i,2])
