import os
import sys
import pandas as pd
import requests
from lxml import html
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import numpy as np
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

"""
Transform data from yaml format to CSV format
"""
def dataTransformer(dir, csvFile):
    df = pd.DataFrame()

    for fn in os.listdir(dir):
        fn = os.path.join(dir, fn)
        #print(fn)
        if os.path.isdir(fn):
            print("What is a folder doing in there ?")
            exit(1)
            continue

        f = open(fn, 'r')
        data = load(f, Loader=Loader)
        
        info = data['info']
        
        if 'India' not in info['teams']:
        	continue

        if not 'city' in info:
            print("City not found in file %s" % (fn))
            continue

        if len(info['dates']) > 1:
            print("Multiple dates in file %s" % (fn))
            continue

        if info['toss']['decision'] == 'bat':
            first_to_bat = info['teams'].index(info['toss']['winner'])
        else:
            first_to_bat = (info['teams'].index(info['toss']['winner']) + 1) % 2
         


        if 'winner' in info['outcome']:
            win_or_draw = 0  #0 = win
        elif info['outcome']['result'] == 'tie':
            win_or_draw = 1 # 1 = tie
        else:
            win_or_draw = 2 # 2 = no result
         
        innings = data['innings']

        indian_players = set()
        other_players = set()

        #print (len(innings))
        for inning in innings:
        	inning = inning[next(iter(inning))]
        	if inning['team'] == 'India':
        		for delivery_dict in inning['deliveries']:
        			delivery = delivery_dict[next(iter(delivery_dict))]

        			batsman = delivery['batsman']
        			bowler = delivery['bowler']
        			non_striker = delivery['non_striker']

        			indian_players.add(batsman)
        			indian_players.add(non_striker)
        			other_players.add(bowler)
        	else:
        		for delivery_dict in inning['deliveries']:
        			delivery = delivery_dict[next(iter(delivery_dict))]

        			batsman = delivery['batsman']
        			bowler = delivery['bowler']
        			non_striker = delivery['non_striker']

        			other_players.add(batsman)
        			other_players.add(non_striker)
        			indian_players.add(bowler)

        final_dict = {'Year'  : [info['dates'][0].year],
                     'Month' : [info['dates'][0].month], 
                     'Day' : [info['dates'][0].day], 
                     'City' : [info['city']], 
                     'Venue' : [info['venue']], 
                     'FirstTeam' : [info['teams'][0]], 
                     'SecondTeam' : [info['teams'][1]], 
                     'FirstToBat' : [first_to_bat], 
                     'Result' : [win_or_draw],
                     'Winner' : [info['teams'].index(info['outcome']['winner'])] if win_or_draw == 0 else [0]
                   }

        if len(indian_players) > 11 or len(other_players) > 11:
        	print("Number of players > 11")
        	continue

        i = 1
        for playa in indian_players:
            final_dict['IndianPlayer%d' % (i)] = playa
            i += 1
        while i <= 11:
            final_dict['IndianPlayer%d' % (i)] = ''
            i += 1

        i = 1
        for playa in other_players:
            final_dict['OppositePlayer%d' % (i)] = playa
            i += 1
        while i <= 11:
            final_dict['OppositePlayer%d' % (i)] = ''
            i += 1

        this_row = pd.DataFrame(final_dict)
        df = pd.concat([df, this_row])

    #df = fetchInfo(df)
    #df = fixIrregularities(df)

    #print(city_set)
    df.to_csv(csvFile, index=False)
    print("Output written to %s" % (csvFile))
    return df

#-----------------------------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":

    df = dataTransformer('data', 'with_players.csv')
