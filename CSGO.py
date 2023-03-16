import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

csData = pd.read_csv('D:/CSGO_Stats/CSGO_data.csv')
#csData = pd.read_csv('C:/Users/Akshansh dhiman/Desktop/CSGO_Stats/CSGO_data_M.csv')

maps = ["Mirage","Dust II","Inferno","Cache","Train","Nuke"]
palete = ["orange","red","green","cyan","blue","magenta"]
csData["K/D"] =  (csData[csData["Deaths"]!=0]["Kills"])/(csData[csData["Deaths"]!=0]["Deaths"])
csData["K/D"].fillna(1,inplace = True)
csData["(K+A)/D"] = (csData[csData["Deaths"]!=0]["Kills"]+csData[csData["Deaths"]!=0]["Assists"])/(csData[csData["Deaths"]!=0]["Deaths"])
csData["(K+A)/D"].fillna(1,inplace =True)
csData.loc[csData["Result"]=="Win","Result"] = 1
csData.loc[csData["Result"]=="Lost","Result"] = 0
csData.loc[csData["Result"]=="Tie","Result"] = 0.5

csDataNT = csData.drop(columns = ["Day","Month","Year"])
stats = csDataNT.describe()
stats = stats.T
stats["IQR"] = stats["75%"]-stats["25%"]



def pie(attribute):    
    attributePMap = []
    for m in maps:
        count = csData[csData["Map"]==m][attribute].sum()
        attributePMap.append(count)
    plt.figure()
    plt.title("Total {} Per Map".format(attribute))
    plt.pie(attributePMap,labels = maps)

matchCount = []
for m in maps:
    count = csData[csData["Map"]==m]["Kills"].count()
    matchCount.append(count)
plt.figure()
plt.title("Total Matches Played Per Map")
plt.pie(matchCount,labels = maps)
plt.show()    

def avgAtMapBar(attributes):
    plt.figure()
    for attribute in attributes:
        matchCount = []
        for m in maps:
            count = csData[csData["Map"]==m]["Kills"].count()
            matchCount.append(count)

        attributePMap = []
        for m in maps:
            count = csData[csData["Map"]==m][attribute].sum()
            attributePMap.append(count)
    
        avgAttributePerMap = []
        i = -1
        for m in maps:
            i = i+1;
            avgAttributePerMap.append(attributePMap[i]/matchCount[i])
        if len(attributes)==1:
            plt.title("Average {} Per Map".format(attribute))
        else:
            plt.title("Average Per Map")
        alp = 1/len(attributes)
        plt.bar(x=maps,height = avgAttributePerMap,color = ["orange","red","green","cyan","blue","magenta"],alpha = alp)
        plt.show()



def attDisHist(attributes,maps,bins=7):
    plt.figure()
    i =-1
    alp = 1/(len(attributes)*len(maps))
    for attribute in attributes:
        if len(attributes)>1: 
            i = i+1    
        for m in maps:
            i = i+1
            if(len(attributes)==1 and len(maps)==1):
                plt.title("{} Distribution on {}".format(attribute,m))
            elif(len(attributes)==1 and len(maps)!=1):
                plt.title("{} Distribution on different Maps".format(attribute))
            elif(len(attributes)!=1 and len(maps)==1):
                plt.title("Distribution on {}".format(m))
            else:
                plt.title("Distribution on different Maps")
            csData[csData["Map"]==m][attribute].hist(bins = bins,color = palete[i],alpha =alp)
            plt.show()
    
binsize = 8
binavg = []
for i in range(0,len(csData["K/D"])+1,binsize):
    binsum = 0
    for j in range(0,binsize):
        try:
            binsum = binsum + csData["K/D"][i+j]
        except:
            pass
    binavg.append(binsum/binsize)
plt.figure()
plt.title("KD Trend (BinSize = {})".format(binsize))
binavg.reverse()
poly = PolynomialFeatures(degree = 2)
X = poly.fit_transform(np.array([x for x in range(0,len(binavg))]).reshape(-1,1))
regressor = LinearRegression()
regressor.fit(X,binavg)
predKills = regressor.predict(X)
plt.plot([x for x in range(0,len(binavg))],predKills,color = "r",)
plt.scatter([x for x in range(0,len(binavg))],binavg)
plt.plot([x for x in range(0,len(binavg))],binavg,color = "green")
plt.xlabel("Time")
plt.ylabel("Average KD")
plt.legend(["True","Predicted","Points"])
plt.show()

kDW = list(csData["K/D"])
kDWO = []
for i in kDW:
    if(i<stats.at["K/D","75%"]+1.5*stats.at["K/D","IQR"] and i>stats.at["K/D","25%"]-1.5*stats.at["K/D","IQR"]):
        kDWO.append(i)
    else:
        #kDWO.append(stats.at["K/D","mean"])
        pass
plt.figure()
plt.subplot(1,2,1)
plt.boxplot(kDW)
plt.subplot(1,2,2)
plt.boxplot(kDWO)
plt.show()

plt.figure()
plt.scatter(csData["Kills"],csData["Mvp's"])
plt.title("Scatter Plot between two attributes")
plt.xlabel("Attribute 1")
plt.ylabel("Attribute 2")
plt.show()
np.cov(csData["Kills"],csData["Deaths"])
np.corrcoef(csData["Kills"],csData["Result"])

corMat = csDataNT.drop(columns = ["Map"]).corr(method = "pearson")
a = 0
for i in csData["Result"]:
    if(type(i)==float):
        a = a+1







