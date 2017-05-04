from sys import argv
import pandas as pd
inp=open(argv[1],'r')

out1 = argv[1][:argv[1].find(".")]

out=open(out1+"_shift.xyz",'w')

out2=open(out1+"_negat_shift.xyz", 'w')

lines=inp.readlines()
coords = []
atoms = 0
hatms = ['CU','MG','ZN','NA','CA']

x=3
y=4
z=2

for line in lines:
  if line.startswith('HETATM'):
    woorden=line.split()
    for atm in hatms:
      if woorden[11]==atm:
        atoms+=1
        coord=[woorden[11],float(woorden[6])+x,float(woorden[7])+y,float(woorden[8])+z]
        coords.append(coord)

A=[]
        
print("      %d" % atoms, file= out)
print(" ", file= out)
#print("\n", file = out)
for i in coords:
   print("%s %8.3f %8.3f %8.3f" % (i[0], i[1], i[2], i[3]), file = out)
   A.append(pd.Series([i[0], i[1], i[2], i[3]],index=['name', 'x', 'y', 'z']))
   
V=pd.Series(['vec',x,y,z],index=['name','x','y', 'z'])

for i in range(len(coords)):
  Xi=A[i].values[1]-2*V.values[1]
  Yi=A[i].values[2]-2*V.values[2]
  Zi=A[i].values[3]-2*V.values[3]
  print("%s %8.3f %8.3f %8.3f" % (A[i].values[0],Xi,Yi,Zi), file = out2)

#A = pd.Series(coords, index=range(len(coords)))

#vector = ['',x,y,z]

#cvector= []

#for l in range(len(coords)):
#  cvector.append(vector)

#B = pd.Series(cvector,index=range(len(coords)))

#newS = A + B

#print(newS)

#C=[5,4]
#D=[7,1]

#print([C[i]+D[i] for i in range(len(C))])

inp.close()
out.close()
out2.close()