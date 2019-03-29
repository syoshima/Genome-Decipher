# -*- coding: utf-8 -*-
"""
Original: PCA and K-Means Decipher Genome (Alexander N. Gorban and Andrei Y. Zinovyev)
Created on Sun Feb 10 20:40:47 2019
@author: Samy Abud Yoshima
PCA
Take the whole dataset consisting of d-dimensional samples ignoring the class labels
Compute the d-dimensional mean vector (i.e., the means for every dimension of the whole dataset)
Compute the scatter matrix (alternatively, the covariance matrix) of the whole data set
Compute eigenvectors (ee1,ee2,...,eed) and corresponding eigenvalues (λλ1,λλ2,...,λλd)
Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix WW(where every column represents an eigenvector)
Use this d×k eigenvector matrix to transform the samples onto the new subspace. 
This can be summarized by the mathematical equation: yy=WWT×xx (where xx is a d×1-dimensional vector representing one sample, and yy is the transformed k×1-dimensional sample in the new subspace.)
k-Means


"""
import math
import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt
import re
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
from matplotlib.mlab import PCA as mlabPCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Capture and clean data
fname = "ccrescentus.txt"
gene = open(fname)
seq = []
Z = []
for line in gene:
    line = line.rstrip()
    seq.append(line)
Z = ''.join(seq)
X = list()
for e in Z:
    if e not in X:
        X.append(e)
X = sorted(X)
print('Elements in gene: ',X)
tot_mg = 0
W = []
Y = list()
for e in X:
     W = float(Z.count(e))
     Y.append(W)
     print(W)
     tot_mg = float(tot_mg) + float(Z.count(e))
print(round(tot_mg))
# Segregate data with 300 letters per row (or index)
wid =  300
data = [Z[i:i+wid] for i in range(0, len(Z), wid)]
with open("data_gene.txt", "w") as output:
    output.write(str(data))

# Featurize and create dataset, using Regular Expression, for each possible word size
# For each word size (from 1 to 4 letters), create all possible combinations of letters
#X1
X1 = [''.join(x1) for x1 in X]
Y1  = pd.DataFrame(columns = X1)
for i in range(len(data)):
    txt =  data[i]
    lst1 = (re.sub("(.{1})", "\\1 ",txt))
    wordlist1 = lst1.split()
    wordfreq1 = []
    for w in X1:
        wordfreq1.append(float(wordlist1.count(w)))
    Y1.loc[i] = wordfreq1
m1 = Y1.mean()
s1 = Y1.std()
data1n = (Y1 - m1) / s1
cov1 = data1n.cov()
eig_val1, eig_vec1 = np.linalg.eig(cov1) # eigenvectors and eigenvalues from the cov matrix
eig_pairs1 = [(np.abs(eig_val1[i]), eig_vec1[:,i]) for i in range(len(eig_val1))]# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs1.sort(key=lambda x: x[0], reverse=True) # Sort the (eigenvalue, eigenvector) tuples from high to low
matrix_w1 = np.hstack((eig_pairs1[0][1].reshape(len(X1),1), eig_pairs1[1][1].reshape(len(X1),1)))
MW_1 = matrix_w1.T
T1 = MW_1.dot(data1n.T)

#X2
X2 = [''.join((x1,x2)) for x1 in X for x2 in X]
Y2  = pd.DataFrame(columns = X2)
for i in range(len(data)):
    txt =  data[i]
    lst2 = (re.sub("(.{2})", "\\1 ",txt))
    wordlist2 = lst2.split()
    wordfreq2 = []
    for w in X2:
        wordfreq2.append(float(wordlist2.count(w)))
    Y2.loc[i] = wordfreq2
m2 = Y2.mean()
s2 = Y2.std()
data2n = (Y2 - m2) / s2
cov2 = data2n.cov()
eig_val2, eig_vec2 = np.linalg.eig(cov2) # eigenvectors and eigenvalues from the cov matrix
eig_pairs2 = [(np.abs(eig_val2[i]), eig_vec2[:,i]) for i in range(len(eig_val2))]# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs2.sort(key=lambda x: x[0], reverse=True) # Sort the (eigenvalue, eigenvector) tuples from high to low
matrix_w2 = np.hstack((eig_pairs2[0][1].reshape(len(X2),1), eig_pairs2[1][1].reshape(len(X2),1)))
MW2 = matrix_w2.T
T2 = MW2.dot(data2n.T)

#X3
X3  = [''.join((x1,x2,x3)) for x1 in X for x2 in X for x3 in X]
Y3  = pd.DataFrame(columns = X3)
for i in range(len(data)):
    txt3 =  data[i]
    lst3 = (re.sub("(.{3})", "\\1 ",txt3))
    wordlist3 = lst3.split()
    wordfreq3 = []
    for w in X3:
        wordfreq3.append(float(wordlist3.count(w)))
    Y3.loc[i] = wordfreq3
m3 = Y3.mean()
s3 = Y3.std()
data3n = ((Y3 - m3) / s3)
cov3 = data3n.cov()
eig_val3, eig_vec3 = np.linalg.eig(cov3) # eigenvectors and eigenvalues from the cov matrix
#eigvec3 = [eig_vec3[:,i].reshape(1,len(X3)).T for i in range(len(eig_val3))]
eig_pairs3 = [(np.abs(eig_val3[i]), eig_vec3[:,i]) for i in range(len(eig_val3))]# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs3.sort(key=lambda x: x[0], reverse=True)
matrix_w3 = np.hstack((eig_pairs3[0][1].reshape(len(X3),1), eig_pairs3[1][1].reshape(len(X3),1)))
T3 = matrix_w3.T.dot(data3n.T)

#X4
X4 = [''.join((x1,x2,x3,x4)) for x1 in X for x2 in X for x3 in X for x4 in X]
Y4  = pd.DataFrame(columns = X4)
for i in range(len(data)):
    txt4 =  data[i]
    lst4 = (re.sub("(.{4})", "\\1 ",txt4))
    wordlist4 = lst4.split()
    wordfreq4 = []
    for w in X4:
        wordfreq4.append(float(wordlist4.count(w)))
    Y4.loc[i] = wordfreq4
m4 = Y4.mean()
s4 = Y4.std()
data4n = (Y4 - m4) / s4
cov4 = data4n.cov()
eig_val4, eig_vec4 = np.linalg.eig(cov4) # eigenvectors and eigenvalues from the cov matrix
eig_pairs4 = [(np.abs(eig_val4[i]), eig_vec4[:,i]) for i in range(len(eig_val4))]# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs4.sort(key=lambda x: x[0], reverse=True) # Sort the (eigenvalue, eigenvector) tuples from high to low
matrix_w4 = np.hstack((eig_pairs4[0][1].reshape(len(X4),1), eig_pairs4[1][1].reshape(len(X4),1)))
MW4 = matrix_w4.T
T4 = MW4.dot(data4n.T)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(T1[0,:], T1[1,:], 'o', markersize=1, color='blue', alpha=0.5, label='class1')
#plt.xlabel('x_values')
#plt.ylabel('y_values')
#plt.legend()
ax1.set_title('(m=1)')
ax2.plot(T2[0,:], T2[1,:], 'o', markersize=1, color='blue', alpha=0.5, label='class1')
ax2.set_title('(m=2)')
ax3.plot(T3[0,:], T3[1,:], 'o', markersize=1, color='blue', alpha=0.5, label='class1')
ax3.set_title('T(m=3)')
ax4.plot(T4[0,:], T4[1,:], 'o', markersize=1, color='blue', alpha=0.5, label='class1')
ax4.set_title('(m=4)')
fig.savefig("PCA_m1to4.png")   
plt.show()

fig=plt.figure(1)
T4t =  pd.DataFrame(data = T4.T, columns = ['PC1', 'PC2'])
kmeans4 = KMeans(n_clusters=3)
kmeans4.fit(T4t)
y_kmeans4 = kmeans4.predict(T4t)
plt.scatter(T4t['PC1'], T4t['PC2'], c=y_kmeans4, s=10, cmap='viridis')
centers4 = kmeans4.cluster_centers_
plt.scatter(centers4[:, 0], centers4[:, 1], c='black', s=100, alpha=1);
fig.savefig("KMeans_m4K7.png")
fig=plt.figure(3)

T3t =  pd.DataFrame(data = T3.T, columns = ['PC1', 'PC2'])
kmeans3 = KMeans(n_clusters=n_cl, random_state=n_cl)
kmeans3.fit(T3t)
y_kmeans3 = kmeans3.predict(T3t)
cnames = ['k','r','g','b','m','c','y']
newcmp = ListedColormap(cnames)
labels = [i for i in range(0,n_cl)]
plt.scatter(T3t['PC1'], T3t['PC2'], c=y_kmeans3, s=10, cmap=newcmp)
centers3 = kmeans3.cluster_centers_
n_cl=len(centers3)
plt.scatter(centers3[:, 0], centers3[:, 1], c='black', s=100, alpha=1);
for label, x, y in zip(labels, centers3[:, 0], centers3[:, 1]):
    plt.annotate(label,xy=(x, y),xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
fig.savefig("KMeans_m3K7.png")   
labels3 = kmeans3.labels_

print('The structure of 3 non-verlapping word letters confirm the relevance of codons.')
print('Genetic material with relevant code information begins and ends with codons')
#Task list
# Gen_Browser
# we will show 100 fragments in the detailed view
fig = plt.figure(4)
I = n_cl
n = 50
plt.ylim(0,n)
t = 0
for i,cl in zip(data[0:n],labels3[0:n]):
    if cl == 0:     
        plt.text(0,t,data[0:n],fontsize=4, color=cnames[0])
    elif cl == 1:     
        plt.text(0,t,data[0:n],fontsize=5, color=cnames[1])
    elif cl == 2:     
        plt.text(0,t,data[0:n],fontsize=6, color=cnames[2])
    elif cl == 3:     
        plt.text(0,t,data[0:n],fontsize=7, color=cnames[3])
    elif cl == 4:     
        plt.text(0,t,data[0:n],fontsize=6, color=cnames[4])
    elif cl == 5:     
        plt.text(0,t,data[0:n],fontsize=5, color=cnames[5])
    elif cl == 6:     
        plt.text(0,t,data[0:n],fontsize=4, color=cnames[6])
    t = t+1
fig.savefig("clustergene.png")
plt.show()
"""
    for z in range(0,I):
        if cl == z :
            plt.text(0,t,data[0:n],fontsize=(3+z), color=cnames[z])

1) Find the correct cluster for informational genetic material, where the correct
triplet distribution (probably) will contain the lowest frequency of the stop
codons TAA, TAG and TGA, the specialized codons). Stop codon can appear only once
in a gene because it terminates its transcription.
"""
collist = ['taa','tag','tga']
SCC = np.array(Y3[collist].sum(axis=1))
#for k in range(0,6):
 #   SCC[k] = [(d) for d,cl in zip(SCC,labels3) if cl == k]

for i in range(0,I):
    globals()['SSC'+str(i)] = [d for d,cl in zip(SCC,labels3) if cl == i] 
XCC = [i for i in range(0,I)]
SCCx = [np.sum(globals()['SSC'+str(i)]) for i in range(0,I)]
fig = plt.figure(5)
plt.bar(XCC, SCCx, color=cnames)
plt.ylabel('Score fo stop codons (taa,tga,tag)')
plt.xlabel('Clusters(labels)')
plt.title('Correct Cluster Shift \n lowest frequency of stop codons')
fig.savefig("clustercorrect.png")   
"""
2) Measure information content for every phase
We can calculate the information value of this triplet
distribution I(F) for each afragment F . Is the information of fragments in
the cluster with a correct shift significantly different from the information
of fragments in other clusters? Is the mean information value in the cluster
with a correct shift significantly different from the mean information value
of fragments in other clusters? Could you verify the hypothesis that in the
cluster with a correct shift the mean information value is higher than in other
clusters?
I = sum([f(ijk) * ln(f(ijk)) / p(i).p(j).p(k))])
fijk is the frequency of triplet ijk.
pi is a frequency of letter i
"""
Y1 = np.array(Y1)
p_i = Y1.sum(axis=1)
p_ii = Y1/p_i[1]
p_a = p_ii[:,[0]]
p_c = p_ii[:,[1]]
p_g = p_ii[:,[2]]
p_t = p_ii[:,[3]]
p_a = np.mean(p_a)
p_c = np.mean(p_c)
p_g = np.mean(p_g)
p_t = np.mean(p_t)

r1 = [list(x) for x in X3]
r1 = np.array(r1)
r1[r1 == 'a'] = 0
r1[r1 == 'c'] = 1
r1[r1 == 'g'] = 2
r1[r1 == 't'] = 3
r1 = r1.astype(float)
r1[r1 == 0] = p_a
r1[r1 == 1] = p_c
r1[r1 == 2] = p_g
r1[r1 == 3] = p_t
r2 = r1[:,0]*r1[:,1]*r1[:,2]
r3 = np.sum(Y3,axis=0)
r4 = Y3/r3
r5 = r4/r2
r6 = r4*np.log(r5)
r6.fillna(0,inplace=True)
IR = np.sum(r6,axis=1)
#for k in range(0,6):
 #   IR[k] = [i for i,cl in zip(IR,labels3) if cl == k]
for t in range(0,I):
    globals()['IR'+str(t)] = [i for i,cl in zip(IR,labels3) if cl == t] 
IRm = [np.mean(globals()['IR'+str(i)]) for i in range(0,I)]
IRl = [len(globals()['IR'+str(i)]) for i in range(0,I)]
IRv = [np.var(globals()['IR'+str(i)]) for i in range(0,I)]
IRs = [np.std(globals()['IR'+str(i)]) for i in range(0,I)]
fig = plt.figure(7)
plt.bar(XCC, IRm, color=cnames)
plt.ylabel('Mean IR of each cluster \n (collection of fragments with same label)')
plt.xlabel('Clusters(labels)')
plt.grid()
plt.title('Correct Cluster Shift \n Max. Informatio Ratio of triplets')
fig.savefig("clustercorrect2.png")   
plt.show()

# Hypothesis testing for significance of Information Ratio
# (check variance of 4 first cluster in Info)
I = I -1
Info = np.array([IRm,IRl,IRv,IRs])
Info.sort(axis=1)
CCCu = np.mean(Info[0,0:(I-2)])
CCC = max(IRm)
# Calculate degress of freedom v
v = ((Info[2,I]/Info[1,I]+np.mean(Info[2,0:(I-2)])/np.mean(Info[1,0:(I-2)]))**2)/((Info[2,I]/Info[1,I])**2/(Info[1,I]+1)+(np.mean(Info[2,0:(I-2)])/np.mean(Info[1,0:(I-2)]))**2/(np.mean(Info[1,0:(I-2)])+1))-2
#print(v, "degress of freedom")
dif_stdev = (Info[2,I]/Info[1,I]+np.mean(Info[2,0:(I-2)])/np.mean(Info[1,0:(I-2)]))**0.5
print("Dif St.Dev",dif_stdev)
t = np.abs(CCC - CCCu)/dif_stdev
## Compare with the critical t-value
#Degrees of freedom
df = v
#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)
print("t = " + str(t))
print("p = " + str(2*p))
print("We can reject the hypothesis that the Information Ratio \n of the correct cluster and the average IR of the other clusters \n are equal at ",round(100*(1-2*p),2)-0.1,"% significance level",'\n') 
#Note that we multiply the p value by 2 because its a two tail t-test   
