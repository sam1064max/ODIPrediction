import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import ensemble
import os
import sys
import requests
from lxml import html
from sklearn.metrics import accuracy_score
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#citymap to fetch weather data
weather_citymap = {('Jaipur', 'Sawai Mansingh Stadium'): 'VIJP', ('Centurion', 'SuperSport Park'): 'FAWK', ('Gwalior', 'Captain Roop Singh Stadium'): 42361, ('Edinburgh', 'Grange Cricket Club Ground, Raeburn Place'): 'EDI', ('Mumbai', 'Wankhede Stadium'): 'BOM', ('Bloemfontein', 'OUTsurance Oval'): 'FABL', ('Dharamsala', 'Himachal Pradesh Cricket Association Stadium'): 42062, ('Cardiff', 'Sophia Gardens'): 'EGFF', ('Mount Maunganui', 'Bay Oval'): 93186, ('Christchurch', 'Hagley Oval'): 'NZCH', ('Birmingham', 'Edgbaston'): 'EGBB', ('Potchefstroom', 'Senwes Park'): 'PCF', ('Canberra', 'Manuka Oval'): 'CBR', ('Christchurch', 'Jade Stadium'): 'NZCH', ('Jamaica', 'Sabina Park, Kingston'): 'MKJP', ('Paarl', 'Boland Park'): 'FACT', ('Dominica', 'Windsor Park, Roseau'): 'DCF', ('Indore', 'Maharani Usharaje Trust Cricket Ground'): 'IDR', ('Visakhapatnam', 'Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium'): 'VTZ', ('Hambantota', 'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa'): 'VCRI', ('Amstelveen', 'VRA Ground'): 'EHAM', ('Skating and Curling Club', 'Toronto Cricket'): 'CYTZ', ('Darwin', 'Marrara Cricket Ground'): 'DRW', ('Kimberley', 'De Beers Diamond Oval'): 'FAKM', ('St Vincent', 'Arnos Vale Ground, Kingstown'): 'TVSV', ('Hyderabad', 'Rajiv Gandhi International Stadium, Uppal'): 'HYD', ('Karachi', 'National Stadium'): 'OPKC', ('Glasgow', 'Titwood'): 'EGPF', ('East London', 'Buffalo Park'): 'FAEL', ('Johannesburg', 'New Wanderers Stadium'): 'FAOR', ('Durban', 'Kingsmead'): 'FALE', ('Nagpur', 'Vidarbha Cricket Association Ground'): 'VANP', ('Nottingham', 'Trent Bridge'): 'EGNX', ('Colombo', 'R Premadasa Stadium'): 'VCBI', ('Rajkot', 'Madhavrao Scindia Cricket Ground'): 42737, ('Pune', 'Maharashtra Cricket Association Stadium'): 43063, ('Nairobi', 'Gymkhana Club Ground'): 'HKNW', ('Sind', 'Niaz Stadium, Hyderabad'): 41764, ('Nelson', 'Saxton Oval'): 'NSN', ('Khulna', 'Sheikh Abu Naser Stadium'): 41947, ('St Kitts', 'Warner Park, Basseterre'): 'TKPK', ('Bangalore', 'M Chinnaswamy Stadium'): 'VOBG', ('Vadodara', 'Indian Petrochemicals Corporation Limited Sports Complex Ground'): 42748, ('Dharmasala', 'Himachal Pradesh Cricket Association Stadium'): 42062, ('Barbados', 'Kensington Oval, Bridgetown'): 'TBPB', ('Dublin', 'The Village, Malahide'): 'DUB', ('Ayr', 'Cambusdoon New Ground'): 'EGPK', ('Benoni', 'Willowmoore Park'): 'JNB', ('Chandigarh', 'Punjab Cricket Association Stadium, Mohali'): 42107, ('Perth', 'Western Australia Cricket Association Ground'): 'YPPH', ('Dubai', 'ICC Global Cricket Academy'): 'OMDB', ('Port Elizabeth', "St George's Park"): 'FAPE', ('Nairobi', 'Jaffery Sports Club Ground'): 'HKNW', ('London', "Lord's"): 'EGLL', ('Bloemfontein', 'Mangaung Oval'): 'FABL', ('Indore', 'Holkar Cricket Stadium'): 42754, ('Dubai', 'ICC Academy'): 'OMDB', ('Bloemfontein', 'Goodyear Park'): 'FABL', ('Trinidad', "Queen's Park Oval, Port of Spain"): 'TTPP', ('Chennai', 'MA Chidambaram Stadium, Chepauk'): 'MAA', ('Antigua', 'Sir Vivian Richards Stadium, North Sound'): 'TAPA', ('Potchefstroom', 'Sedgars Park'): 'PCF', ('Cape Town', 'Newlands'): 'FACT', ('Nagpur', 'Vidarbha Cricket Association Stadium, Jamtha'): 'VANP', ('Napier', 'McLean Park'): 93373, ('Hobart', 'Bellerive Oval'): 'YMHB', ('Colombo', 'Sinhalese Sports Club Ground'): 'CMB', ('Faisalabad', 'Iqbal Stadium'): 'OPFA', ('Kanpur', 'Green Park'): 'VILK', ('Cuttack', 'Barabati Stadium'): 42970, ('Dunedin', 'University Oval'): 93891, ('Bristol', 'County Ground'): 'EGGD', ('Kochi', 'Nehru Stadium'): 'VOCI', ('Nairobi', 'Ruaraka Sports Club Ground'): 'HKNW', ('Mirpur', 'Shere Bangla National Stadium'): 41923, ('Colombo', 'P Saravanamuttu Stadium'): 'VCBI', ('Chandigarh', 'Sector 16 Stadium'): 'IXC', ('Bloemfontein', 'Chevrolet Park'): 'FABL', ('Grenada', "National Cricket Stadium, St George's"): 'TGPY', ('Lahore', 'Gaddafi Stadium'): 'OPLA', ('Guwahati', 'Nehru Stadium'): 42410, ('Chittagong', 'Zahur Ahmed Chowdhury Stadium'): 'CGP', ('Fatullah', 'Khan Shaheb Osman Ali Stadium'): 41923, ('Aberdeen', 'Mannofield Park'): 'EGPD', ('Abu Dhabi', 'Sheikh Zayed Stadium'): 'OMAA', ('Toronto', 'Maple Leaf North-West Ground'): 'CYTZ', ('Visakhapatnam', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium'): 'VTZ', ('Lincoln', 'Bert Sutcliffe Oval'): 'NZCH', ('Bogra', 'Shaheed Chandu Stadium'): 41883, ('Hamilton', 'Seddon Park'): 93186, ('Kuala Lumpur', 'Kinrara Academy Oval'): 'WMKK', ('Manchester', 'Old Trafford'): 'EGCC', ('Rajkot', 'Saurashtra Cricket Association Stadium'): 'VARK', ('Ranchi', 'JSCA International Stadium Complex'): 42701, ('Auckland', 'Eden Park'): 'AKL', ('Chester-le-Street', 'Riverside Ground'): 'EGNT', ('Vadodara', 'Reliance Stadium'): 'VABO', ('Guyana', 'Providence Stadium'): 'SYCJ', ('Kolkata', 'Eden Gardens'): 'VECC', ('St Lucia', 'Beausejour Stadium, Gros Islet'): 'TLPC', ('Wellington', 'Basin Reserve'): 'NZWN', ('Christchurch', 'AMI Stadium'): 'NZCH', ('London', 'Kennington Oval'): 'EGLL', ('Ahmedabad', 'Sardar Patel Stadium, Motera'): 'AMD', ('Dublin', 'Clontarf Cricket Club Ground'): 'EIDW', ('Delhi', 'Feroz Shah Kotla'): 'VIDP', ('Leeds', 'Headingley'): 'EGNM', ('Southampton', 'The Rose Bowl'): 'EGHI', ('Bulawayo', 'Queens Sports Club'): 'FVBU', ('Chittagong', 'Zohur Ahmed Chowdhury Stadium'): 'VGEG', ('Whangarei', 'Cobham Oval (New)'): 'WRE', ('Brisbane', 'Brisbane Cricket Ground, Woolloongabba'): 'YBBN', ('Margao', 'Nehru Stadium, Fatorda'): 'VAGO', ('Mumbai', 'Brabourne Stadium'): 'BOM', ('King City', 'Maple Leaf North-West Ground'): 'KKIC', ('Belfast', 'Civil Service Cricket Club, Stormont'): 'EGAA', ('Wellington', 'Westpac Stadium'): 'NZWN'}

"""
Fetch other needed information,
For now, we are only fetching weather info
"""
def fetchInfo(frame):
    frame1 = pd.DataFrame()

    for (index, row) in frame.iterrows():
        res = fetchWeatherForDateAndCity(row['City'], row ['Venue'], row['Year'], row['Month'], row['Day'])
        if res is not None:
            (temperature, humidity, wind_speed, dew_point) = res

            mapping = {'Year'  : [row['Year']],
                         'Month' : [row['Month']], 
                         'Day' : [row['Day']], 
                         'City' : [row['City']], 
                         'Venue' : [row['Venue']], 
                         'FirstTeam' : [row['FirstTeam']], 
                         'SecondTeam' : [row['SecondTeam']], 
                         'FirstToBat' : [row['FirstToBat']], 
                         'Winner' : [row['Winner']],
                         'Temp' : [temperature],
                         'Humidity' : [humidity],
                         'WindSpeed' : [wind_speed],
                         'DewPoint' : [dew_point]
                       }

            for i in range(1, 12):
                mapping['IndianPlayer%d' % (i)] = row['IndianPlayer%d' % (i)]
                mapping['OppositePlayer%d' % (i)] = row['OppositePlayer%d' % (i)]
            
            new_row = pd.DataFrame(mapping)

            frame1 = pd.concat([frame1, new_row])

            frame1.to_csv("newTraining.csv", index=False)
    return frame1

"""
Fetches weather info for provided city and venue, year, month and day from
Wunderground.com
We manually found (city, venue) to nearest airport mapping
and weather data is extracted for that airport for the date of the match.
The mapping is stored at the bottom of the file in weather_citymap
"""
def fetchWeatherForDateAndCity(city, venue, year, month, day):
    if (city, venue) not in weather_citymap:
        print('%s and %s not in weather_citymap' % (city, venue))
        return None
    else:
        airport_code = weather_citymap[(city, venue)]
        if type(airport_code) is str:
            page = requests.get('https://www.wunderground.com/history/airport/%s/%d/%d/%d/DailyHistory.html' % (airport_code, year, month, day))
        else:
            page = requests.get('https://www.wunderground.com/history/wmo/%d/%d/%d/%d/DailyHistory.html' % (airport_code, year, month, day))
        
        tree = html.fromstring(page.content)

        try:
            label = tree.xpath('//*[@id="historyTable"]/tbody/tr[2]/td[1]/span/text()')[0]
            i = 2
            while label != 'Mean Temperature':
                i += 1
                label = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[1]/span/text()' % (i))
                if len(label) > 0:
                    label = label[0]
            temperature = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[2]/span/span[1]/text()' % (i))[0]

            label = tree.xpath('//*[@id="historyTable"]/tbody/tr[2]/td[1]/span/text()')[0]
            i = 2
            while label != 'Wind Speed':
                i += 1
                label = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[1]/span/text()' % (i))
                if len(label) > 0:
                    label = label[0]
            wind_speed = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[2]/span/span[1]/text()' % (i))[0]

            label = tree.xpath('//*[@id="historyTable"]/tbody/tr[2]/td[1]/span/text()')[0]
            i = 2
            while label != 'Dew Point':
                i += 1
                label = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[1]/span/text()' % (i))
                if len(label) > 0:
                    label = label[0]

            dew_point = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[2]/span/span[1]/text()' % (i))[0]

            label = tree.xpath('//*[@id="historyTable"]/tbody/tr[2]/td[1]/span/text()')[0]
            i = 2
            while label != 'Average Humidity':
                i += 1
                label = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[1]/span/text()' % (i))
                if len(label) > 0:
                    label = label[0]

            humidity = tree.xpath('//*[@id="historyTable"]/tbody/tr[%d]/td[2]/text()' % (i))[0]
        except Exception:
            print("Exception caught")
            return None

        return (temperature, humidity, wind_speed, dew_point)

"""
Transform data from yaml format to CSV format
"""
def dataTransformer(dir, csvFile, parse_yaml_files = False):
    if parse_yaml_files:
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
            
            #Only consider India
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
                            
            if 'winner' not in info['outcome']:
                continue # We only want win-lose test cases
            
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

            this_row = pd.DataFrame({'Year'  : [info['dates'][0].year],
                                     'Month' : [info['dates'][0].month], 
                                     'Day' : [info['dates'][0].day], 
                                     'City' : [info['city']], 
                                     'Venue' : [info['venue']], 
                                     'FirstTeam' : [info['teams'][0]], 
                                     'SecondTeam' : [info['teams'][1]], 
                                     'FirstToBat' : [first_to_bat], 
                                     'Winner' : [info['teams'].index(info['outcome']['winner'])]
                                   })

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

        df = fetchInfo(df)
    else:
        df = pd.read_csv(csvFile, header=0)

    df.to_csv(csvFile, index=False)
    print("Output written to %s" % (csvFile))
    return df

#-----------------------------------------------------------------------------------------------------------------------------------------#

# Generating Features

df = pd.read_csv("training.csv")
test_df = pd.read_csv("test_case.csv")

train_df_len = len(df)

#Joining the test and trining data to make one-hot encoding easier
df = pd.concat([df, test_df])

X= pd.DataFrame()
X=pd.concat([X,pd.get_dummies(df['Venue'],prefix='Venue')],axis=1)

X=pd.concat([X,pd.get_dummies(df['SecondTeam'],prefix='SecondTeam')],axis=1)

# Make a 0-1 encoding for all indian players
India=pd.DataFrame()
India=pd.concat([India,df['IndianPlayer1']],axis=1)
India=pd.concat([India,df['IndianPlayer2']],axis=1)
India=pd.concat([India,df['IndianPlayer3']],axis=1)
India=pd.concat([India,df['IndianPlayer4']],axis=1)
India=pd.concat([India,df['IndianPlayer5']],axis=1)
India=pd.concat([India,df['IndianPlayer6']],axis=1)
India=pd.concat([India,df['IndianPlayer7']],axis=1)
India=pd.concat([India,df['IndianPlayer8']],axis=1)
India=pd.concat([India,df['IndianPlayer9']],axis=1)
India=pd.concat([India,df['IndianPlayer10']],axis=1)
India=pd.concat([India,df['IndianPlayer11']],axis=1)
India=pd.concat([pd.get_dummies(India[col]) for col in India], axis=1, keys=India.columns)

# Make a 0-1 encoding for all opposition players
Opposite=pd.DataFrame()
Opposite=pd.concat([Opposite,df['OppositePlayer1']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer2']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer3']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer4']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer5']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer6']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer7']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer8']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer9']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer10']],axis=1)
Opposite=pd.concat([Opposite,df['OppositePlayer11']],axis=1)
Opposite=pd.concat([pd.get_dummies(Opposite[col]) for col in Opposite], axis=1, keys=Opposite.columns)

# The win rate of india in that particular stadium

X['Stadium Win Rate']=df['Stadium Win Rate']
X['FirstToBat']=df['FirstToBat']
X['Home Field Advantage']=df['Home Field Advantage']
X['WinLose ratio bat or bowl first against team']=df['WinLose ratio bat or bowl first against team']

X['DewPoint']=df['DewPoint']
X['Humidity']=df['Humidity']
X['WindSpeed']=df['WindSpeed']
X['Temp']=df['Temp']

X=pd.concat([X,India],axis=1)
X=pd.concat([X,Opposite],axis=1)

y=df['Winner']

X = X.values
y = y.values

# Here, we split the training and test data which we merged above

X_train = X[:(train_df_len), :]
Y_train = y[:(train_df_len)]

X_test = X[(train_df_len):, :]
Y_test = y[(train_df_len):]

np.random.seed(0)

'''
for i in np.linspace(1,20000,num=100):
    clf=SVC(kernel='rbf',C=int(i),gamma=0.0001)
    scores = cross_validation.cross_val_score(clf, X, y, cv=3)
    print "Accuracy:" ,(np.mean(np.sqrt(abs(scores))))
    print i
'''

clf=SVC(kernel='rbf', C=5000, gamma=0.0001)
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=3)
print("Mean 3-fold cross validation accuracy:" ,(np.mean(np.sqrt(abs(scores)))))

clf.fit(X_train, Y_train)

print("\nRunning test cases...")

Y_pred = clf.predict(X_test)

for i in range(len(Y_pred)):
    winner_team = test_df['SecondTeam'].iloc[i] if Y_pred[i] == 1 else 'India'
    print("Case %d : Team %s wins" % (i + 1, winner_team))
