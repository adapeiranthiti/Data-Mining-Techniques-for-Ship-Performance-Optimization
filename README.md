# Path Optimization Based on Weather Conditions

*This project is an attempt to approach the issue of ship route optimization based on weather conditions. Data analysis techniques and deep learning algorithms were used to achieve our goal.*

## Contents

- Libraries Used
- Data Collection
- Data Preprocessing
- Heatmap and Corellation
- Anomaly Detection

### Libraries Used 

Below I will link the documentations of the libraries used for the project. For further reading you can open the documentation links.

- Pandas [documentation](https://pandas.pydata.org/docs/index.html)
- NumPy [documentation](https://numpy.org/doc/)
- Math [documentation](https://docs.python.org/3/library/math.html)
- Matplotlib [documentation](https://matplotlib.org/stable/index.html)
- GeoPy [documentation](https://geopy.readthedocs.io/)
- TensorFlow [documentation](https://www.tensorflow.org/api_docs)
- Seaborn [documentation](https://seaborn.pydata.org/)
- Sklearn [documentation](https://scikit-learn.org/)
- Pyproj [documentation](https://pyproj4.github.io/pyproj/)
- Json [documentation](https://docs.python.org/3/library/json.html)
- Requests [documentation](https://realpython.com/python-requests/)
- SciPy [documentation](https://docs.scipy.org/doc/)


### Data Collection

Fistly, I initialized some variables that we are going to use later in the code
```py
geodesic = pyproj.Geod(ellps='WGS84') #Calculating the geodesic distances
R = 6378.1  #Radius of the Earth in km
tf.random.set_seed(69) #TensorFlow generator
```

Mounting the drive to access the csv file
```py
from google.colab import drive
drive.mount('/content/drive')
```

Loading the dataset from the drive to python as a dataframe
```py
df = pd.read_csv('/content/drive/4_EO4EU.csv')
```

### Data Preprocessing 

Filtering the dataset, excluding the faulty data
```py
df = df[(df['stw']>=5) & (df['stw']<=17) & (df['foc']>0) & (df['wind_s']>0) & (df['rpm']>0) & (df['power']>0)]
```

Checking the dataset, for duplicates and missing values
```py
print('Number of missing values = %d' % (df.isna().sum()))
print('Number of duplicates = %d' % (dups.sum()))
```

Applying the harvensine formula to the lat and lon columns of the dataset
```py
def sp_harvensine(lat, lon, degrees = True):
  r = 6371 #Earth's radius (km)
  if degrees:
    lat, lon = map(radians, [lat, lon]) #Converting demical degrees to radians

  #Computing the harvensine function
  a = sin(lat/2) **2 + cos(lat) * sin(lon/2) **2
  d = 2 * r * asin(sqrt(a))

  return d

df['lat_lon'] = df.apply(lambda x: sp_harvensine(x['lat'], x['lon']), axis = 1) 
```

### Heatmap and Correlation

Computing the correlation pairs with all the data and with foc only
```py
correlation_pairs_all = []
correlation_pairs_foc = []

for i in range(len(correlation_matrix.columns)):
  for j in range(i+1, len(correlation_matrix.columns)):
    correlation_pairs_all.append((correlation_matrix.columns[i], correlation_matrix.columns[j], abs(correlation_matrix.iloc[i,j])))

correlation_pairs_all = correlation_pairs_all.sort(key = lambda x: abs(x[2]), reverse = True)

for i in range(len(correlation_matrix.columns)):
  for j in range(i+1, len(correlation_matrix.columns)):
    if correlation_matrix.columns[i] == 'foc' or correlation_matrix.columns[j] == 'foc':
      correlation_pairs_foc.append((correlation_matrix.columns[i], correlation_matrix.columns[j], abs(correlation_matrix.iloc[i,j])))

correlation_pairs_foc = correlation_pairs_foc.sort(key = lambda x: abs(x[2]), reverse = True)
```

Checking correlation with wind and wave
```py
#Correlation experiment with foc - rpm - wind_d
print(df[['head', 'wind_d', 'foc']].corr()) #Printing the correlations between these 3 features

df2 = df[['head', 'wind_d', 'foc', 'rpm']] #Creating a dataframe only with these 4 features from df

df2['headwind'] = df2.apply(lambda x: (x['head'] - x['wind_d']), axis = 1) #Creating a new column from the features 'head' and 'wind_d'
df2 = df2.drop(['head', 'wind_d'], axis = 1)

correlation_matrix = df2.corr()

#Correlation experiment with foc - rpm - wind_s
print(df[['wind_s', 'foc']].corr()) #Printing the correlations between these 3 features

df2 = df[['wind_s', 'foc', 'rpm']] #Creating a dataframe only with these 4 features from df

correlation_matrix = df2.corr()

#Correlation experiment with foc - rpm - wave_d
print(df[['head', 'wave_d', 'foc']].corr()) #Printing the correlations between these 3 features

df2 = df[['head', 'wave_d', 'foc', 'rpm']] #Creating a dataframe only with these 4 features from df

df2['headwave'] = df2.apply(lambda x: (x['head'] - x['wave_d']), axis = 1) #Creating a new column from the features 'head' and 'wave_d'
df2 = df2.drop(['head', 'wave_d'], axis = 1)

correlation_matrix = df2.corr()

#Correlation experiment with foc - rpm - wave_h
print(df[['wave_h', 'foc']].corr()) #Printing the correlations between these 3 features

df2 = df[['wave_h', 'foc', 'rpm']] #Creating a dataframe only with these 4 features from df

correlation_matrix = df2.corr()
```

Checking correlation with all the features including lat and lon
```py
df2 = df.drop(['dt'], axis=1)

for col in df2.columns:
	df2[col] = df2[col][(np.abs(stats.zscore(df2[col])) < 3)] #Adding to the col column of df2 the values of the zscores that are < 3

df2 = df2.dropna()

correlation_matrix = df2.corr()
```

### Anomaly Detection

Initializing some variables that we are going to use later in the project
```py
n_most_important = 7 #The number of the attributes we want to see the importance of
rows_for_feature_sel = 500 #The number of the rows we want to take from the selected features
site_id = 4
lstm = True #LSTM usage
vessel = None
```

Tried to connect to an API to access weather data (unsuccessfully)
```py
'''def wind_speed(latlons, dts):  #All values returned were NaN
	payload = json.dumps({
		"dts": dts, #Time stamps
		"latlons": latlons, #Geographical positions
	})

	headers = {
		'Content-Type': 'application/json',
	}
	#192.168.201.25:5007
	#195.97.37.253:5007
	url = "https://195.97.37.253:5007/weatherServiceList/" #Taking weather info for specific geological positions
	response = requests.request("POST", url, headers=headers, data=payload, verify=False)

  # print(response.json())
	df = pd.DataFrame(response.json())[['wind_speed_kn', 'wind_direction_deg']]

	return df'''
```

Initializing some variables essensial for the rest of the project
```py
df_sel = df #The selected df will be the initial df
df_sel = df_sel.head(rows_for_feature_sel) #The selected df will contain only the first 500 rows
df_rf_x = df_sel.drop(['dt', 'foc', 'sovg', 'stemHour', 'lat', 'lon', 'comb_d', 'comb_h', 'curr_s', 'curr_d', 'trim'], axis = 1) #These columns are not needed
df_rf_x = df_rf_x.apply(lambda x: x.astype(float)) #Make all the attributes floats
df_rf_y = df_sel['foc'].values #Creating a dataframe with foc values
```

Creating a feature selection function to find the n 7 most important features, for the foc prediction
```py
def feature_selection(x, y, features, alg):
  most_important = []
  #Use Random Forest when 'alg' = 'rf'
  if alg == 'rf':
    rf = RandomForestRegressor(random_state = 0)

    rf.fit(x,y) #Training the data x and y calculating the significance of them using feature_importances_
    f_i = list(zip(features, rf.feature_importances_)) 
    f_i.sort(key = lambda x: x[1])
    print("Selected features with RFR:\n", f_i)

    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])

    most_important = [k[0] for k in f_i]

  #Use Lasso when 'alg' != 'rf'
  else:
    lar = Lasso()
    lar.fit(x,y) #Training the data x and y calculating the coefficients of the features
    coeffs = lar.coef_ 
    most_important = np.abs(coeffs)

    f_i = list(zip(features, most_important))
    f_i.sort(key = lambda x: x[1])
    print("Selected features with LASSO:\n", f_i)

    plt.barh([x[0] for x in f_i], [x[1] for x in f_i]) 

  return most_important
```

Computing the 7 most important features using:

- Random Forest Regressor (RFR)
```py
cols_fs = feature_selection(df_rf_x.values, df_rf_y, df_rf_x.columns, 'rf')

cols_mi = cols_fs[::-1][:n_most_important]
cols_rest = cols_fs[::-1][n_most_important:]

print("{} most important features for FOC estimation: {}".format(n_most_important, cols_mi))
```
- LASSO
```py
cols_fs = feature_selection(df_rf_x.values, df_rf_y, df_rf_x.columns, 'ls')

cols_mi = cols_fs[::-1][:n_most_important]
cols_rest = cols_fs[::-1][n_most_important:]

print("{} most important features for FOC estimation: {}".format(n_most_important, cols_mi))
```

Initializing the 7 most important attributes to features column
```py
features_model = cols_mi #Initializing the 7 most important attributes to features model
features_model.append('foc') #Adding to the list the attribute "foc"
features_model.append('rpm') #Adding to the list the attribute "rpm"
```





