import numpy as np

lijst = []
para = True
while para:
	invoer = input('Give number: ')
	if invoer=='quit':
		para = False
	else:
		lijst.append(float(invoer))

blist = np.array(lijst)

print('average: ', round(np.average(blist),3))

print('standard deviation: ', round(np.std(blist),3))

#gem = sum(lijst)/len(lijst)
#print ('average: ', round(gem,3))

#devia = 0
#for i in lijst:
#	devia += (i-gem)**2

#stan_dev = (devia/len(lijst))**0.5

#print ('standard deviation: ', round(stan_dev,3))
	