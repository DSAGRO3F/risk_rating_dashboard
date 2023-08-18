
import streamlit as st
import traceback
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt


# Page initialization
st.set_page_config(page_title="Home credit dashboard", layout='wide')

# Set page title
st.title('Home credit dashboard')

# Set connection
session = requests.Session()


# Set script to retrieve X_test data from @app.route('/get_X_test_data/', methods=['GET', 'POST'])

def get_X_test_dataset(session, url_X_test):
    try:
        # retry = Retry(connect=3, backoff_factor=0.5)
        # adapter = HTTPAdapter(max_retries=retry)
        # session.mount('http://', adapter)
        # session.mount('https://', adapter)

        result = session.get(url_X_test)
        result = result.json()
        return result

    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()
        return {}

#base_url = ' http://127.0.0.1:5000'
base_url = 'https://github.com/DSAGRO3F/risk_rating_api.git'
end_point = '/get_X_test_data'

liste = get_X_test_dataset(session, url_X_test = base_url + end_point)
print('liste:{}'.format(liste[0:5]))
make_choice = st.sidebar.selectbox('Select current file:', liste)


customer_id = make_choice
print('customer_id: {}'.format(customer_id))

l_keys = []
l_values = []
l_feat_imp = []
l_feat_imp_values = []


def status(session, url_get_inp_data):
    try:
        result = session.get(url_get_inp_data)
        result = result.json()
        print('result: {}'.format(result))
        return result

    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()
        return {}


end_point_get_inp_data = '/get_input_data/' + customer_id
url_get_inp_data = base_url + end_point_get_inp_data

print('url_get_inp_data: {}'.format(url_get_inp_data))


# 1. Call status function.
result_inp_data = status(session, url_get_inp_data=url_get_inp_data)
print(len(result_inp_data))
print(type(result_inp_data))

# 2. Assign outputs to variables.
customer_id = result_inp_data['customer_id']
y_pred = result_inp_data['y_pred'][0]
client = result_inp_data['df_client']
clt_info = result_inp_data['df_clt_feat']

# 3. Convert to dataframes
df_client = pd.read_json(client)
df_clt_info = pd.read_json(clt_info)
#print(df_clt_info[0:2])


# Call feature importance api
end_point_feat_imp = '/feat_imp/' + customer_id
print(base_url + end_point_feat_imp)


def fetch(session, url_feat_imp):
    try:
        results = session.get(url_feat_imp)
        results = results.json()
        #print('results: {}'.format(results))

        f_imp = results['feature_importances']
        #print('f_imp: {}'.format(f_imp))
        df_elig_f_imp = results['eligible_clients']
        df_client_f_imp = results['NOT_eligible_client']

        for key in f_imp:
            l_keys.append(f_imp[key])
        #print("cles {}".format(l_keys))

        nb_ele = len(l_keys)
        for i in range(nb_ele):

            for j in range(len(l_keys[i])):
                l_values.append(l_keys[i][str(j)])

        for i in range(len(l_values)):
            nb_values = len(l_values)
            value = l_values[i]

            if i <= (nb_values - 1)//2:
                l_feat_imp.append(value)

            if i > (nb_values - 1)//2:
                l_feat_imp_values.append(value)

        #print("1___: {}".format(l_feat_imp))
        #print("2___: {}".format(l_feat_imp_values))

        dic = {'feat_imp': l_feat_imp, 'feat_imp_values': l_feat_imp_values}
        df = pd.DataFrame(dic)
        df['description'] = ['Normalized score from external data source',
                             'Normalized score from external data source',
                             'Family status of the client',
                             'Flag if client permanent address does not match work address (1=different, 0=same, at city level)',
                             'Gender of the client', 'Level of highest education the client achieved',
                             'Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application)',
                             'How many days before the application the person started current employment',
                             'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                             'How many days before the application the person started current employment']

        #print('df___{}'.format(df))

        d = {'df': df, 'df_elig_f_imp': df_elig_f_imp, 'df_client_f_imp': df_client_f_imp}

        return d

    except Exception:
        return {}

# 1. Call fetch function et assign outputs to variables.
results = fetch(session, url_feat_imp = base_url + end_point_feat_imp)


# 2. Assign outputs to variables.
results_feat_imp = results['df']
results_elig_f_imp = results['df_elig_f_imp']
#print('results_elig_f_imp: {}'.format(results_elig_f_imp))

results_client_f_imp = results['df_client_f_imp']
print('results_client_f_imp: {}'.format(results_client_f_imp))

# 3. Convert to dataframes
df_elig_f_imp = pd.read_json(results['df_elig_f_imp'])
df_client_f_imp = pd.read_json(results['df_client_f_imp'])
print("df_client_f_imp: {}".format(df_client_f_imp))

# 4. Preparing feature importances dataframe prior to display
df_feat_imp = results_feat_imp.loc[:, ['feat_imp', 'feat_imp_values']]

# 5. Make a scatter of 'df_elig_f_imp' and 'df_client_f_imp'.

# 5.1. Assign values to new tag feature.
df_elig_f_imp['eligible'] = 1
df_client_f_imp['eligible'] = 0

# 5.2. Concatenate dataframes.
df_both = pd.concat([df_elig_f_imp, df_client_f_imp], axis=0)
df_both.reset_index(inplace=True, drop=True)
# print(df_both.shape)
# print(df_both)

# 5.3. Make PCA() to display eligible clients sample versus current on scatter plot.
pca = PCA(n_components=2)
matrix = pca.fit_transform(df_both.values)
new_df = pd.DataFrame(matrix, columns=['pc_0', 'pc_1'])
new_df['eligible'] = df_both['eligible']
# print("new: {}".format(new_df))

# 5.4. Display results
x= new_df['pc_0']
y = new_df['pc_1']
c = new_df['eligible']

fig, ax = plt.subplots()
ax.scatter(x=x, y=y, c=c)
ax.set_title('Curr. file compared to eligible ones')
ax.set_xlabel("pc_0")
ax.set_ylabel("pc_1")
plt.grid()


# Organize dashboard page
with st.container():
    st.title(':black[Features related to current file]')
    st.write('Getting data related to current file...')
    st.write('Current SK_ID_CURR file Id: ', customer_id)
    st.dataframe(df_clt_info)

with st.container():
    st.title(':black[Main features, status tables comparison: eligible clients and current file]')
    col_1, col_2 = st.columns(2, gap="large")
    with col_1:
        st.subheader(':green[Main features, eligible clients sample]')
        col_1 = st.columns(1)
        st.dataframe(df_elig_f_imp)
        st.subheader(':red[Main features, current file]')
        st.dataframe(df_client_f_imp)

    with col_2:
        st.subheader('How is home credit client request ?')
        st.write('Current SK_ID_CURR file Id: ', customer_id)
        if y_pred == float(0):
            st.write('Answer to client request: ', ':red[Credit not accepted]')
        else:
            st.write('Answer to client request: ', ':green[Credit not accepted]')

        st.pyplot(fig)

with st.container():
    col_1, col_2 = st.columns(2, gap="large")
    with col_1:
        st.subheader('Most important features to consider.')
        st.write('Most important features are criteria that most characterize ability to not default credit.')
        st.dataframe(df_feat_imp)
        #st.write(result_feat_imp)

    with col_2:
        st.subheader('Display most important features...')
        st.bar_chart(df_feat_imp, x='feat_imp', y='feat_imp_values', width=0, height=0)
        #fig = px.bar(df_feat_imp, 'feat_imp', 'feat_imp_values')
        #st.plotly_chart(fig, use_container_width=True)
        #st.write(fig)




