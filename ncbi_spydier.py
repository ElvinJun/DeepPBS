from Bio import Entrez
import os
Entrez.email = 'xxxxxxxxxxx@qq.com'  # always tell who you are
# handle = Entrez.egquery(term="E.coli")
# record = Entrez.read(handle)
# for row in record["eGQueryResult"]:
#     if row["DbName"]=="pubmed":
#         print row["Count"]
handle = Entrez.esearch(db="pubmed", term="growth phase" , retmax=500000)
record = Entrez.read(handle)
idlist = record["IdList"]
list3_2= idlist
print(len(list3_2))

Entrez.email = 'xxxxxxxxxxx@qq.com'  # always tell who you are
# handle = Entrez.egquery(term="promoter")
# record = Entrez.read(handle)
# for row in record["eGQueryResult"]:
#     if row["DbName"]=="pubmed":
#         print row["Count"]
handle = Entrez.esearch(db="pubmed", term="stress response", retmax=500000)
record = Entrez.read(handle)
idlist = record["IdList"]
list3_3 = idlist
print(len(list3_3))

Entrez.email = 'xxxxxxxxxxx@qq.com'  # always tell who you are
# handle = Entrez.egquery(term="stationary phase")
# record = Entrez.read(handle)
# for row in record["eGQueryResult"]:
#     if row["DbName"]=="pubmed":
#         print row["Count"]

handle = Entrez.esearch(db="pubmed", term="acid response", retmax=500000)
record = Entrez.read(handle)
idlist = record["IdList"]
list3_4 = idlist
print(len(list3_4))


handle = Entrez.esearch(db="pubmed", term="pH response", retmax=5000000 )
record = Entrez.read(handle)
idlist = record["IdList"]
list3_5 = idlist
print(len(list3_5))

f3_2 = open(os.path.join(os.getcwd(),'growth phase.txt'), 'w')
f3_3 = open(os.path.join(os.getcwd(),'stress response.txt'), 'w')
f3_4 = open(os.path.join(os.getcwd(),'acid response.txt'), 'w')
f3_5 = open(os.path.join(os.getcwd(),'pH response.txt'), 'w')

for i in list3_2:
    f3_2.write(i + ' \n')
for i in list3_3:
    f3_3.write(i + ' \n')
for i in list3_4:
    f3_4.write(i + ' \n')
for i in list3_5:
    f3_5.write(i + '\n ')

f3_2.close()
f3_3.close()
f3_4.close()
f3_5.close()

