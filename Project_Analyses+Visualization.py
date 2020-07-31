# -*- coding: utf-8 -*-
"""
Created on Sun May 17 04:50:45 2020

@author: DELL
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
#import re
def sentimentfunc(tweet):
    try:
        return TextBlob(tweet).sentiment.polarity
    except:
        return None
SNP30 = pd.read_csv("SNP30.csv")
#print(SNP30.index)


# ---- exploratory analysis ----


#1.	Total Number of tweets

print("Total number of tweets", SNP30.text.nunique())
#2.	Number of retweets

print("Total number of retweets",SNP30.text.str.startswith("RT").sum())
#3.	Total number of unique users
print("Total number of unique users",SNP30.user_id.nunique())
#4.	Number of tweets containing a URL.
print("Total number of tweets containing a URL",SNP30.text.str.contains("https:").sum())
#5.	Number of tweets that are replies (starting with “@”)
print("Total number of tweets that are replies",SNP30.text.str.startswith("@").sum())
# assumption that "RT @" is a retweet not a reply. If it was a reply, the query would change to str.contains instead of str.startswith

#---- hourly tweets analysis ----

#name_hours = ['1','2','3','4', '5' , '6', '7', '8', '9', '10', '11' , '12' , '13' , '14' , '15' , '16' , '17' , '18' , '19' , '20' , '21', '22' , '23' , '00']
SNP30['Hour'].plot.hist(bins = 24)
plt.title("Hour wise tweet distribution")

plt.show()


#---- yearly tweet analysis ----

#name_years = ['2007', '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018', '2019']
SNP30['Year'].plot.hist(bins = 12)
plt.title("Year wise tweet distribution")

plt.show()


#---- analysis on sources ----

iPhone = SNP30[SNP30.Mobile_App_iPhone == True]
#print(iPhone.head())
#print(iPhone.dtypes)
Android = SNP30[SNP30.Mobile_App_Android == True]
MobileWeb = SNP30[SNP30.Mobile_Web == True]
Desktop = SNP30[SNP30.Desktop_Web == True]
TweetDeck = SNP30[SNP30.TweetDeck == True]
#Others = SNP30[SNP30.Mobile_App_iPhone == False and SNP30.Mobile_App_Android == False and SNP30.TweetDeck == False and SNP30.Desktop_Web == False and SNP30.Mobile_Web == False]
uniqueusers = [iPhone.user_id.nunique(), Android.user_id.nunique(),MobileWeb.user_id.nunique(), Desktop.user_id.nunique(),TweetDeck.user_id.nunique()]
name_uniqueusers = ['iPhone','Android','MobileWeb','Desktop','TweetDeck']

#sns.barplot(name_uniqueusers, uniqueusers)
#plt.title("Unique users per platform")
#plt.show()

tweetsperuniqueuser = [(iPhone.id_str.count()/iPhone.user_id.nunique()),(Android.id_str.count()/Android.user_id.nunique()),(MobileWeb.id_str.count()/MobileWeb.user_id.nunique()), (Desktop.id_str.count()/Desktop.user_id.nunique()),(TweetDeck.id_str.count()/TweetDeck.user_id.nunique())]
#sns.barplot(name_uniqueusers, tweetsperuniqueuser)
#plt.title("Average tweets per user from each platform")
#plt.show()


#print(SNP30.isnull().sum())
#print(Desktop.describe(include = 'all'))
#print(iPhone.describe())




#---- text cleaning ----

SNP30['full_tweet_senti'] = SNP30.full_tweet.str.replace('[^0-9a-zA-Z]+',' ')

#---- analysis on user description ----
nodescr = SNP30[SNP30.user_description.isnull() == True ]
descr = SNP30[SNP30.user_description.isnull() == False ]
print(nodescr["user_followers_count"].describe())
print(descr["user_followers_count"].describe())

y = SNP30['favorite_count']
xforreg = SNP30.filter(['quote_count','reply_count','retweet_count', 'user_verified','user_friends_count', 'user_followers_count','user_listed_count','user_favourites_count','language_EN'])
x = xforreg

regr = LinearRegression()
regr.fit(x,y)
print(regr.score(x,y))
#regressions give score of how much these factors in line 59 affected the favourite count




#---- sentiment analysis on tweets----

SNP30['senti_score'] = SNP30['full_tweet'].apply(sentimentfunc)
#help taken from 2 stackoverflow answers on this

print(SNP30['senti_score'].describe())


# ---- mention of each company

SNP30['AXP'] = SNP30['text'].str.contains('AXP | American Express Co | @AmericanExpress').astype(int)
SNP30['AAPL'] = SNP30['text'].str.contains('AAPL | Apple | @Apple').astype(int)
SNP30['DIS'] = SNP30['text'].str.contains('DIS | The Walt Disney Company | @theDIS').astype(int)
SNP30['RTX'] = SNP30['text'].str.contains('RTX | Raytheon Tehcnologies Corporation | @RaytheonTech').astype(int)
SNP30['HD'] = SNP30['text'].str.contains('HD | The Home Depot | @HomeDepot').astype(int)
SNP30['BA'] = SNP30['text'].str.contains('BA | The Boeing Company | @Boeing').astype(int)
SNP30['WMT'] = SNP30['text'].str.contains('WMT | Walmart | @Walmart').astype(int)
SNP30['MMM'] = SNP30['text'].str.contains('MMM | 3M | @3M').astype(int)
SNP30['JNJ'] = SNP30['text'].str.contains('JNJ | Johnson & Johnson | @JNJNews').astype(int)
SNP30['JPM'] = SNP30['text'].str.contains('JPM | J.P. Morgan | @jpmorgan').astype(int)
SNP30['CAT'] = SNP30['text'].str.contains('CAT | CaterpillarInc | @CaterpillarInc').astype(int)
SNP30['GS'] = SNP30['text'].str.contains('GS | Goldman Sachs | @GoldmanSachs').astype(int)
SNP30['MSFT'] = SNP30['text'].str.contains('MSFT | Microsoft | @Microsoft').astype(int)
SNP30['V'] = SNP30['text'].str.contains('V | Visa | @Visa').astype(int)
SNP30['INTC'] = SNP30['text'].str.contains('INTC | Intel | @intel').astype(int)
SNP30['WBA'] = SNP30['text'].str.contains('WBA | Walgreens | @Walgreens').astype(int)
SNP30['IBM'] = SNP30['text'].str.contains(' IBM | International Business Machines | @IBM').astype(int)
SNP30['TRV'] = SNP30['text'].str.contains('TRV |The Travelers Co |@Travelers').astype(int)
SNP30['UNH'] = SNP30['text'].str.contains('UNH | UnitedHealth Group | @UnitedHealthGrp').astype(int)
SNP30['DOW'] = SNP30['text'].str.contains('DOW | DOW Inc | @Dow').astype(int)
SNP30['MRK'] = SNP30['text'].str.contains('MRK | Merck & Co | @Merck').astype(int)
SNP30['VZ'] = SNP30['text'].str.contains('VZ | Verizon Communication | @Verizon').astype(int)
SNP30['PFE'] = SNP30['text'].str.contains('PFE | Pfizer Inc | @Pfizer').astype(int)
SNP30['NKE'] = SNP30['text'].str.contains('NKE | Nike Inc | @NKE').astype(int)
SNP30['PG'] = SNP30['text'].str.contains('PG | The Procter & Gamble Co | @ProcterGamble').astype(int)
SNP30['XOM'] = SNP30['text'].str.contains('XOM | Exxon Mobil Co | @Exxonmobil').astype(int)
SNP30['MCD'] = SNP30['text'].str.contains('MCD | McDonald Co | @McDonaldsCorp').astype(int)
SNP30['CSCO'] = SNP30['text'].str.contains('CSCO | Cisco Systems | @Cisco').astype(int)
SNP30['CVX'] = SNP30['text'].str.contains('CVX | Chevron Corporation | @Chevron').astype(int)
SNP30['KO'] = SNP30['text'].str.contains('KO | The Coca Cola Company | @CocaColaCo').astype(int)

names_companycomp = ['AXP','AAPL','DIS','RTX','HD','BA','WMT','MMM','JNJ','JPM','CAT','GS','MSFT','V','INTC','WBA','IBM','TRV','UNH','DOW','MRK','VZ','PFE','NKE','PG','XOM','MCD','CSCO','CVX','KO']
companycomp = [SNP30.AXP.sum(),SNP30.AAPL.sum(),SNP30.DIS.sum(),SNP30.RTX.sum(),SNP30.HD.sum(),SNP30.BA.sum(),SNP30.WMT.sum(),SNP30.MMM.sum(),SNP30.JNJ.sum(),SNP30.JPM.sum(),SNP30.CAT.sum(),SNP30.GS.sum(),SNP30.MSFT.sum(),SNP30.V.sum(),SNP30.INTC.sum(),SNP30.WBA.sum(),SNP30.IBM.sum(),SNP30.TRV.sum(),SNP30.UNH.sum(),SNP30.DOW.sum(),SNP30.MRK.sum(),SNP30.VZ.sum(),SNP30.PFE.sum(),SNP30.NKE.sum(),SNP30.PG.sum(),SNP30.XOM.sum(),SNP30.MCD.sum(),SNP30.CSCO.sum(),SNP30.CVX.sum(),SNP30.KO.sum()]
sns.barplot(names_companycomp,companycomp)
plt.xticks(rotation = 90)

plt.show()
#print(SNP30.AXP.sum())

SNP30.to_csv("SNP30analysis.csv")

print(['a','b','c']+[1,2,3])
