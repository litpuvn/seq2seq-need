from noaa_api_v2 import NOAAData

api_token = "your token"
data = NOAAData(api_token)

categories = data.data_categories(locationid='FIPS:37', sortfield='name')

for i in categories:
    print(i)