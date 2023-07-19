
import pandas as pd
from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
import dateutil
import datetime
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

import tabpy_client
connection = tabpy_client.Client('http://localhost:9004/')

def rfm_clustering(_arg1, _arg2, _arg3, _arg4):
    import pandas as pd
    import numpy as np
    import datetime
    import statsmodels.api as sm
    from scipy import stats

    df_raw = pd.DataFrame({'customerId': _arg1, 'OrderDate': pd.to_datetime(_arg2), 'OrderID': _arg3, 'currencyAmount': _arg4})
    # Filter out NA, null, NaN, and values less than 1
    df = df_raw.dropna().query("customerId.notnull() and OrderDate.notnull() and OrderID.notnull() and currencyAmount.notnull() and currencyAmount >= 1")

    max_date = max(df.OrderDate) + datetime.timedelta(days=1)
    customers = df.groupby(['customerId']).agg({
        'OrderDate': lambda x: (max_date - x.max()).days,
        'OrderID': 'count',
        'currencyAmount': 'sum'}).reset_index()
  
    # Rename columns
    customers.rename(columns={'OrderDate': 'Recency',
                              'OrderID': 'Frequency',
                              'currencyAmount': 'MonetaryValue'}, inplace=True)
    
    # Set the Numbers
    customers_fix = pd.DataFrame()
    
    # Check for constant values before applying transformations
    if customers['Recency'].nunique() > 1:
        customers_fix['Recency'] = stats.boxcox(customers['Recency'])[0]
    else:
        customers_fix['Recency'] = customers['Recency']
    
    if customers['Frequency'].nunique() > 1:
        customers_fix['Frequency'] = stats.boxcox(customers['Frequency'])[0]
    else:
        customers_fix['Frequency'] = customers['Frequency']
    
    if customers['MonetaryValue'].nunique() > 1:
        customers_fix['MonetaryValue'] = pd.Series(np.cbrt(customers['MonetaryValue'])).values
    else:
        customers_fix['MonetaryValue'] = customers['MonetaryValue']

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(customers_fix)
    customers_normalized = scaler.transform(customers_fix)

    from sklearn.cluster import KMeans

    # Check if the number of samples is sufficient for the desired number of clusters
    if len(customers_normalized) >= 2:
        # Find optimal number of clusters using the Elbow method
        sse = {}
        for k in range(1, 30):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(customers_normalized)
            sse[k] = kmeans.inertia_

        # Find the number of clusters based on the Elbow method
        # Identify the "elbow" point in the SSE plot
        elbow_k = 0
        prev_sse = float('inf')
        for k, sse_value in sse.items():
            if k > 1:
                sse_diff = prev_sse - sse_value
                if sse_diff < 0.1 * prev_sse:  # Adjust the threshold as per your preference
                    elbow_k = k - 1
                    break
            prev_sse = sse_value

        # Create the K-means model with the optimal number of clusters
        model = KMeans(n_clusters=elbow_k, random_state=42)
        model.fit(customers_normalized)
        customers["Cluster"] = model.labels_

    # Calculate cluster scores
    cluster_scores = customers.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean'
    }).sum(axis=1)

    # Sort clusters based on scores
    ranked_clusters = cluster_scores.sort_values(ascending=False).index

    # Assign ranks to clusters
    customers['Rank'] = customers['Cluster'].map(lambda x: ranked_clusters.get_loc(x) + 1)

    df_return = pd.merge(df_raw, customers, left_on='customerId', right_on='customerId', how='left')
    return df_return['Rank'].tolist()


connection.deploy('RFM Clustering', rfm_clustering, 'Returns flagging of Customer', override=True)

def sarima_method(_arg1, _arg2, _arg3):
    import pandas as pd
    from pandas import DataFrame
    import numpy as np
    import statsmodels.api as sm
    import dateutil
    import datetime
    import warnings
    warnings.filterwarnings('ignore')

    data = DataFrame({'Period': _arg1, 'Net Sales': _arg2})
    data = data.sort_values(by='Period')
    data['Period'] = pd.to_datetime(data['Period'])

    # Exclude the last 3 months
    data = data[:-_arg3]

    # use for training entire dataset
    data.index = pd.to_datetime(data['Period'])
    data = data.resample('M').mean()

    # create future dataset
    step = dateutil.relativedelta.relativedelta(months=1)
    start = data.index[len(data) - 1] + step
    index = pd.date_range(start, periods=_arg3, freq='M')
    columns = ['Net Sales']
    df = pd.DataFrame(index=index, columns=columns)
    df = df.fillna(0)

    # Fit the model
    fit1 = sm.tsa.statespace.SARIMAX(data['Net Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
    df['Net Sales'] = fit1.forecast(_arg3)
    df = df.fillna(0)
    x = pd.concat([data, df])
    return x['Net Sales'].tolist()

connection.deploy('Seasonal ARIMA Method',sarima_method,'Returns forecast of revenue', override=True)

### Prerequisite for the model
### _arg1 = Date as string --> attr(date)  ; _arg2 = Value Aggregate -->  sum(sales)    _arg3 = Int Parameter --> 3,4,5 
def holts_linear_method(_arg1,_arg2,_arg3):
    import pandas as pd
    from pandas import DataFrame
    import numpy as np
    from statsmodels.tsa.api import Holt
    import warnings
    import dateutil
    warnings.filterwarnings('ignore')
    
    data = DataFrame({'Period': _arg1,'Net Sales': _arg2})
    data = data.sort_values(by = 'Period')
    data['Period'] = pd.to_datetime(data['Period'])
    
    #use for training entire dataset
    data.index = pd.to_datetime(data['Period']) 
    data = data.resample('M').mean()
    data = data[:-(_arg3)]
    
    #create future dataset
    step = dateutil.relativedelta.relativedelta(months=1)
    start = data.index[len(data)-1] + step
    index = pd.date_range(start, periods=_arg3, freq='M')
    columns = ['Net Sales']
    df = pd.DataFrame(index=index, columns=columns)
    df = df.fillna(0)

    #Fit the model
    fit1 = Holt(np.asarray(data['Net Sales'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
    df['Net Sales']=fit1.forecast(_arg3)
    df = df.fillna(0)
    x = pd.concat([data, df])
    return x['Net Sales'].tolist()

connection.deploy('Holt Linear Method',holts_linear_method,'Returns forecast of revenue', override=True)