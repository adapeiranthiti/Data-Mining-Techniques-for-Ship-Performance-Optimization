# Data Mining Techniques for Ship Performance Optimization

*This project is an attempt to approach the issue of ship route optimization based on weather conditions. Data analysis techniques and deep learning algorithms were used to achieve our goal.*

## Author

I'm Adamantia Apeiranthiti, a final year undergraduate student at National and Kapodistrian University of Athens. This project is a part of my thesis with objective in Data Mining Techniques for Ship Performance Optimization.
For this project I used [Google Colaboratory](https://colab.research.google.com). The dataset for this thesis, was used in collaboration with [Danaos Shipping Co. Ltd.](https://www.danaos.com/home/default.aspx). Below, you can learn more about the project. 

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

Creating a baseline model based on neurons
```py
def baseline_model(neurons, n_steps, input_shape, lstm, optimizer):

  #Building the model using an LSTM and some Dense layers
  estimator = keras.Sequential() 
  estimator.add(keras.layers.LSTM(neurons, input_shape = [n_steps, input_shape,],)) if lstm else None #if lstm is True then the layer may be lstm
  estimator.add(keras.layers.Dense(neurons, input_shape = [None, input_shape],))

  if neurons > 5:
    neurons = neurons - 5
    estimator.add(keras.layers.Dense(neurons, ))

    neurons = neurons - 5

    #While the number of the neurons is >= 5, we continuou to add denses to the network and every time we decrease the number of neurons of 5
    while(neurons) >= 5:
      estimator.add(keras.layers.Dense(neurons, ))
      neurons = neurons - 5

    estimator.add(keras.layers.Dense(1, )) #At the final dense layer only one neuron that returnes only one output

    estimator.compile(loss = keras.losses.mean_squared_error, optimizer = optimizer, ) #Loss function
    print(estimator.summary())

    return estimator
```

Creating a function that can predict the next feature of a sequence
```py
def split_sequence(sequence, n_steps):
  x, y = list(), list()

  for i in range(len(sequence)):
    end_ix = i + n_steps #Find the end of this pattern

    if end_ix > len(sequence) - 1:
      break

    seq_x, seq_y = sequence[i:end_ix][:, 0:sequence.shape[1] - 1], sequence[end_ix - 1][sequence.shape[1] - 1] #Gather input and output parts of the pattern
    x.append(seq_x)
    y.append(seq_y)

  return np.array(x), np.array(y)
```

Creating a function that plots the anomaly detection results
```py
def plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id):
  plt.clf()
  fig, ax1 = plt.subplots(figsize = (15,10))
  plt.plot(mae_list_loss)
  plt.plot(mae_list_val_loss)
  plt.title('Model train vs Validation loss MAE: %.2f' % (np.round(math.sqrt(mse), 2)) + "," + str(np.round(math.sqrt(np.mean(history.history['val_loss'])))))
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.grid()

  plt.legend(['train', 'validation'], loc = 'upper right')
  plt.show()
```

Creating a DBSCAN anomaly detection algorithm for both rpm and stw usage
- rpm usage:
```py
def dbscan_outlier_detection(df, eps_in = 0.5):

  df_train = df[features_model].head(60000) #Training to the first 60000 samples
  dbscan = DBSCAN(eps = eps_in, n_jobs = -1) #Initialize the DBSCAN model with specified epsilon and all available CPUs for parallel processing
  model = dbscan.fit(df_train[['rpm', 'foc']]) #Training according to the features rpm and foc
  labels = model.labels_ #Retrieve the cluster labels from the DBSCAN model
  df_train["anomaly_score"] = labels #The cluster labels are saved to the column anomaly_score of the dataframe
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#DBSCAN algorithm training
def dbscan_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = dbscan_outlier_detection(df, eps_in = 0.5) #Calling function to detect and remove outliers in the DataFrame df. Use an epsilon value (eps_in) of 0.5 for neighborhood distance.

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

dbscan_res = dbscan_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

- stw usage:
```py
def dbscan_outlier_detection(df, eps_in = 0.5):

  df_train = df[features_model].head(60000) #Training to the first 60000 samples
  dbscan = DBSCAN(eps = eps_in, n_jobs = -1) #Initialize the DBSCAN model with specified epsilon and all available CPUs for parallel processing
  model = dbscan.fit(df_train[['stw', 'foc']]) #Training according to the features stw and foc
  labels = model.labels_ #Retrieve the cluster labels from the DBSCAN model
  df_train["anomaly_score"] = labels #The cluster labels are saved to the column anomaly_score of the dataframe
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#DBSCAN algorithm training
def dbscan_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = dbscan_outlier_detection(df, eps_in = 0.5) #Calling function to detect and remove outliers in the DataFrame df. Use an epsilon value (eps_in) of 0.5 for neighborhood distance.

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

dbscan_res = dbscan_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

Creating a Isolation Forest anomaly detection algorithm for both rpm and stw usage
- rpm usage:
```py
def isolation_forest_outlier_detection(df, contamination_in = 0.01, n_estimators_in = 100, max_samples_in = 'auto', max_features_in = 1.0, bootstrap_in = False, n_jobs_in = -1, random_state_in = 0, verbose_in = 0):

  df_train = df[features_model].head(60000)  #Training to the first 60000 samples
  isolation_forest = IsolationForest(n_estimators = n_estimators_in, contamination = contamination_in, max_samples = max_samples_in, max_features = max_features_in, bootstrap = bootstrap_in, n_jobs = n_jobs_in, random_state = random_state_in, verbose = verbose_in)  #Initialize the iForest model
  model = isolation_forest.fit(df_train[['rpm', 'foc']]) #Training according to the features rpm and foc
  df_train["anomaly_score"] = model.predict(df_train[['rpm', 'foc']]) #Predicting the anomalies in a new column in the dataset
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#iForest algorithm training
def isolation_forest_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = isollation_forest_detection(df, contamination_in = 0.01, n_estimators_in = 100, max_samples_in = 'auto', max_features_in = 1.0, bootstrap_in = False, n_jobs_in = -1, random_state_in = 0, verbose_in = 0) #Calling function to detect and remove outliers in the DataFrame df

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

i_forest_res = isolation_forest_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

- stw usage:
```py
def isolation_forest_outlier_detection(df, contamination_in = 0.01, n_estimators_in = 100, max_samples_in = 'auto', max_features_in = 1.0, bootstrap_in = False, n_jobs_in = -1, random_state_in = 0, verbose_in = 0):

  df_train = df[features_model].head(60000)  #Training to the first 60000 samples
  isolation_forest = IsolationForest(n_estimators = n_estimators_in, contamination = contamination_in, max_samples = max_samples_in, max_features = max_features_in, bootstrap = bootstrap_in, n_jobs = n_jobs_in, random_state = random_state_in, verbose = verbose_in)  #Initialize the iForest model
  model = isolation_forest.fit(df_train[['stw', 'foc']]) #Training according to the features stw and foc
  df_train["anomaly_score"] = model.predict(df_train[['stw', 'foc']]) #Predicting the anomalies in a new column in the dataset
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#iForest algorithm training
def isolation_forest_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = isollation_forest_detection(df, contamination_in = 0.01, n_estimators_in = 100, max_samples_in = 'auto', max_features_in = 1.0, bootstrap_in = False, n_jobs_in = -1, random_state_in = 0, verbose_in = 0) #Calling function to detect and remove outliers in the DataFrame df

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

i_forest_res = isolation_forest_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

Creating a OCSVM anomaly detection algorithm for both rpm and stw usage
- rpm usage:
```py
def ocsvm_outlier_detection(df, kernel_in='rbf', degree_in=3, gamma_in='scale', nu_in=0.01, shrinking_in=True, cache_size_in=200, verbose_in=False, max_iter_in=-1):

  df_train = df[features_model].head(60000)  #Training to the first 60000 samples
  ocsvm_forest = OneClassSVM(kernel=kernel_in, degree=degree_in, gamma=gamma_in, nu=nu_in, shrinking=shrinking_in, cache_size=cache_size_in, verbose=verbose_in, max_iter=max_iter_in)  #Initialize the ocsvm model
  model = ocsvm.fit(df_train[['rpm', 'foc']]) #Training according to the features rpm and foc
  df_train["anomaly_score"] = model.predict(df_train[['rpm', 'foc']]) #Predicting the anomalies in a new column in the dataset
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#OCSVM algorithm training
def ocsvm_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = isollation_forest_detection(df, kernel_in='rbf', degree_in=3, gamma_in='scale', nu_in=0.01, shrinking_in=True, cache_size_in=200, verbose_in=False, max_iter_in=-1) #Calling function to detect and remove outliers in the DataFrame df

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

ocsvm_res = ocsvm_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

- stw usage:
```py
def ocsvm_outlier_detection(df, kernel_in='rbf', degree_in=3, gamma_in='scale', nu_in=0.01, shrinking_in=True, cache_size_in=200, verbose_in=False, max_iter_in=-1):

  df_train = df[features_model].head(60000)  #Training to the first 60000 samples
  ocsvm_forest = OneClassSVM(kernel=kernel_in, degree=degree_in, gamma=gamma_in, nu=nu_in, shrinking=shrinking_in, cache_size=cache_size_in, verbose=verbose_in, max_iter=max_iter_in)  #Initialize the ocsvm model
  model = ocsvm.fit(df_train[['stw', 'foc']]) #Training according to the features rpm and foc
  df_train["anomaly_score"] = model.predict(df_train[['stw', 'foc']]) #Predicting the anomalies in a new column in the dataset
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#OCSVM algorithm training
def ocsvm_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = isollation_forest_detection(df, kernel_in='rbf', degree_in=3, gamma_in='scale', nu_in=0.01, shrinking_in=True, cache_size_in=200, verbose_in=False, max_iter_in=-1) #Calling function to detect and remove outliers in the DataFrame df

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

ocsvm_res = ocsvm_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

Creating a z-Score anomaly detection algorithm for both rpm and stw usage
- rpm usage:
```py
def z_score_outlier_detection(df, threshold_in = 3):

  df_train = df[features_model].head(60000)  #Training to the first 60000 samples
  normality = df_train[(np.abs(stats.zscore(df_train[['rpm', 'foc']])) < 3).all(axis = 1)] #If the zscore is smaller than 3 then -> normal
  anomalies = df_train[(np.abs(stats.zscore(df_train[['rpm', 'foc']])) > 3).all(axis = 1)] #If the zscore is greater than 3 then -> anomaly
  labels = [0 if x in normality.index else 1 for x in df_train.index] #If normal 0 otherwise 1
  df_train["anomaly_score"] = labels
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#z-Score algorithm training
def z_score_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = z_score_detection(df, 3) #Calling function to detect and remove outliers in the DataFrame df

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

zscore_res = z_score_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

- stw usage:
```py
def z_score_outlier_detection(df, threshold_in = 3):

  df_train = df[features_model].head(60000)  #Training to the first 60000 samples
  normality = df_train[(np.abs(stats.zscore(df_train[['stw', 'foc']])) < 3).all(axis = 1)] #If the zscore is smaller than 3 then -> normal
  anomalies = df_train[(np.abs(stats.zscore(df_train[['stw', 'foc']])) > 3).all(axis = 1)] #If the zscore is greater than 3 then -> anomaly
  labels = [0 if x in normality.index else 1 for x in df_train.index] #If normal 0 otherwise 1
  df_train["anomaly_score"] = labels
  anomalies = df_train[df_train.anomaly_score == -1] #Samples with label -1 we be considered anomalies
  data = df_train.drop(anomalies.index.values, axis = 0) #And they will be dropped from the dataframe
  df_train = data.astype(float).dropna() #Droping the Na values

  return df_train

#z-Score algorithm training
def z_score_training(df, site_id, features_model, lstm, min_speed = None, max_speed = None, ):
  df = z_score_detection(df, 3) #Calling function to detect and remove outliers in the DataFrame df

  df = df[features_model] #The df will have only the selected features

  cleaned_data = df

  cleaned_data = cleaned_data.values[:90000] #Selecting only the first 90000 data

  val_len = int(len(cleaned_data) * 0.1) #Setting validation length to 10% of the total cleaned data

  #Separating the feature matrix x and the target variable y from the cleaned data
  x = cleaned_data[:, :cleaned_data.shape[1] - 1] #x contains all columns except the last one (features)
  y = cleaned_data[:, cleaned_data.shape[1] - 1] #y contains the last column (target)

  #Training model
  tr_len = len(cleaned_data) - val_len #Setting the length of the training model equal to the difference of the validation length from the cleaned model length
  partitions_x = cleaned_data[:tr_len, :cleaned_data.shape[1] - 1] #The first tr_len rows are used for training
  partitions_y = cleaned_data[:tr_len, cleaned_data.shape[1] - 1] #The rest tr_len rowsare used for validation

  #Validation model
  df_val = cleaned_data
  
  #The last val_len rows are split into partitions_x_val (features) and partitions_y_val
  partitions_x_val = df_val[-val_len:, :df_val.shape[1] - 1] 
  partitions_y_val = df_val[-val_len:, df_val.shape[1] - 1]

  val_data = np.array(np.append(partitions_x_val, np.asmatrix([partitions_y_val]).T, axis = 1)) #Combines partitions_x_val and partitions_y_val back into a single array val_data for validation purposes

  n_steps = 15 #Defines the number of time steps for sequence data

  input_shape = partitions_x.shape[1] #Defines the input shape as the number of features in the dataset

  raw_seq = np.array(np.append(x, np.asmatrix([y]).T, axis = 1)) #x and y arrays are concatenated into one array

  #If lstm=True
  if lstm:
    x_lstm, y_lstm = split_sequence(raw_seq, n_steps) #Calls the split_sequence function to split the raw data raw_seq into sequences of length n_steps

  neurons = 15 #Defines the number of the neurons to 15
  estimator = baseline_model(neurons, n_steps, input_shape, lstm, optimizer='Adam') #Calls the baseline model to build the model 

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) #Sets up early stopping to monitor val_loss and stop training if the loss doesn't improve after 4 consecutive epochs

  epochs = 100 #Defines the number of epochs to 100

  #If lstm=True
  if lstm:
    history = estimator.fit(x_lstm, y_lstm, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #Τhe LSTM data x_lstm and y_lstm are used
  else:
    history = estimator.fit(partitions_x, partitions_y, epochs=100, verbose=1, validation_split=0.1, callbacks=[callback],) #partitions_x and partitions_y are used

  #Extracts the loss and validation loss from the training history
  list_loss = history.history['loss']
  list_val_loss = history.history['val_loss']

  #Prints the loss and validation loss during training
  print("Loss: ", list_loss)
  print("Val Loss: ", list_val_loss)

  mse = estimator.evaluate(x_lstm, y_lstm) if lstm else estimator.evaluate(partitions_x, partitions_y) #Evaluates the model on the training data and calculates the mean squared error (MSE)
  #Computes the root mean square (RMS) for both the training loss and validation loss
  loss = str(np.round(math.sqrt(np.mean(list_loss))))
  val_loss = str(np.round(math.sqrt(np.mean(list_val_loss))))
  #Converts the list of training and validation losses to their square root (MAE approximation)
  mae_list_loss = [math.sqrt(x) for x in list_loss]
  mae_list_val_loss = [math.sqrt(x) for x in list_val_loss]
  plot_convergence(mae_list_loss, mae_list_val_loss, loss, val_loss, mse, history, site_id) #Calls plot_convergence to generate and plot the convergence of training and validation losses
  print("\nLoss {} Val Loss {}".format(loss, val_loss)) #Prints the final training and validation loss values after model training
  arch = estimator.to_json() #Serializes the model architecture to a JSON string

  return {"loss": loss, "val_loss": val_loss, 'estimator': estimator, 'epochs' : epochs} #Returns a dictionary containing the final loss, validation loss, trained model (estimator), and the number of epochs

zscore_res = z_score_training(df, site_id, features_model, lstm, min_speed = 9, max_speed = 23, ) #Calls the DBSCAN training function
```

After these experiments, I believe that the algorithm with the highest accuracy is the Z-Score using stw.
