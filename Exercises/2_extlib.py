import pandas as pd

input_string = open("3_given_file_story","r")

lim=70

something = input_string.read()

print(something)

l=0

list=[]

teller=0

bool1 = True

bool2 = False

for c in something:
    bool2 = False
    if bool1:
        if c!="\n":
            list.append(c)
            l = l+1
        else:
            part1 = list
            len_part1 = l
            teller = teller + 1
            list = []
            bool1 = False
            bool2 = True

    if not bool1 and not bool2:
        if c!="\n":
            list.append(c)
            l = l+1
        else:
            part2 = list
#            len_part2 = l - len_part1
            teller = teller + 1

print(part1)
print(part2)
len_part2 = len(part2)
print(len_part2)
print(len(part2))
            
#        if teller == 0:
#            part1 = list
#            len_part1 = l
#            teller = teller + 1
#            list = []
#        if teller == 1:
#            part2 = list
#            len_part2 = l - len_part1
#    l = l+1
        
#print(val)

tot = l-1

something1 = pd.Series(part1, index=range(len_part1)) 

something2 = pd.Series(part2, index=range(len_part2))

#frame = pd.Series(something, index=arange(val))

#k=0

#for s in input_string.read().split("\n")
#for s in something1:
#    if s == "": print
w=0
l = []
#for s in part1:
string1 = ''.join(str(s) for s in part1)

#l=[]
#for s in part2:
string2 = ''.join(str(s) for s in part2)

#string2 = l

#l=[]

#for d in something1.str.split():
for d in string1.split():
    if w + len(d) + 1 <= lim:
        l.append(d)
        w += len(d) + 1
    else:
        print (" ".join(str(x) for x in l))
        l = [d]
        w = len(d)
if (len(l)): print (" ".join(str(x) for x in l)+"\n")

#for s in something2:
#    if s == "": print
w=0
l = []
#for d in something2.str.split():
for d in string2.split():
    if w + len(d) + 1 <= lim:
        l.append(d)
        w += len(d) + 1
    else:
        print (" ".join(str(x) for x in l))
        l = [d]
        w = len(d)
if (len(l)): print (" ".join(str(x) for x in l)+"\n")
    
 