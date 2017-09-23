import json
import urllib2

class PlayerStat:
	access_token = ""
	
	json.load(urllib2.urlopen("url"))


import requests

response = requests.get('http://thedataishere.com',
                         auth=('user', 'password'))
data = response.json()


import requests
from requests_kerberos import HTTPKerberosAuth

response = requests.get('http://thedataishere.com',
                         auth=HTTPKerberosAuth())
data = response.json()


