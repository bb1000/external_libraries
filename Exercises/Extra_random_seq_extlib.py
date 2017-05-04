--

(4) Solve exercise 8 ('random amino acid') of "Filehandling" using pandas: define the aminoacids and their letters in a DataFrame. [A]

--

from sys import argv
import pandas as pd
import random
data = {'amin': [ "Ala" , "Cys", "Asp", "Glu", "Phe", "Gly", "His", "Ile", "Lys", "Leu", "Met", "Asn", "Pro", "Gln", "Arg", "Ser" , "Thr" , "Val" , "Trp", "Tyr" ], 'labels': [ "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"] }

frame = pd.DataFrame(data)

#amin = {"A": "Ala" , "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe" , "G": "Gly", "H": "His", "I": "Ile" , "K": "Lys" , "L": "Leu", "M": "Met","N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg", "S": "Ser" , "T": "Thr" , "V": "Val" , "W": "Trp", "Y": "Tyr"}

#amin1=list(amin.keys())
rnumb = int(argv[1])
nfiles = int(argv[2])

#print(frame['labels'])

for i in range(nfiles):
	out = open('sek%i.fasta'%(i+1), 'w')
	print('> Sequence %d'%(i+1)+': ', file = out)
	sek = []
	for d in range(rnumb):
		sek.append(random.choice(frame['labels']))
	sek = ''.join(sek)
	for i in range(0,len(sek),70):
		print(sek[i:i+70], file = out)
	out.close()
		
