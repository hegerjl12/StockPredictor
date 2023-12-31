import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from deta import Deta
import pickle
import datetime
import matplotlib.pyplot as plt

if 'deta' not in st.session_state:
    st.session_state.deta = Deta(st.secrets['DB_TOKEN'])
if 'spy_db' not in st.session_state:
    st.session_state.spy_db = st.session_state.deta.Base('spy_db')
if 'spy_models' not in st.session_state:
    st.session_state.spy_models = st.session_state.deta.Drive('spy_models')

@st.cache_resource
def connect_database():
    st.session_state.deta = Deta(st.secrets['DB_TOKEN'])
    st.session_state.spy_db = st.session_state.deta.Base('spy_db')
    st.session_state.spy_models = st.session_state.deta.Drive('spy_models')

    #return deta, spy_db, spy_models

def process_data(spy_db, newData_df):

    last_entry = None
    i = 0
    date_string = datetime.date.today()
    list_of_times = ['12:30', '11:30', '10:30', '09:30', '08:30', '07:30', '06:30']


    while last_entry is None:
        candle_string = str(date_string) + "T" + list_of_times[i] + ":00-08:00"
        last_entry = spy_db.get(candle_string)
        if i == 6:
            date_string += datetime.timedelta(days=-1)
        i = (i + 1) % 7

    # trim the upload to just columns we care about
    spy_df = newData_df[
        ['time', 'open', 'high', 'low', 'close', 'Momemtum', 'Slow Pressure', 'Fast Pressure']].copy()

    spy_df['change_close_open'] = spy_df['close'] - spy_df['open']
    spy_df['change_high_open'] = spy_df['high'] - spy_df['open']
    spy_df['change_low_open'] = spy_df['low'] - spy_df['open']

    # add column for the deltas for momentum, sp, fp
    m_delta = [0]
    sp_delta = [0]
    fp_delta = [0]

    for i in range(len(spy_df['Momemtum'])):

        if i < len(spy_df['Momemtum']) - 1:
            m_delta.append(spy_df.loc[i + 1, 'Momemtum'] - spy_df.loc[i, 'Momemtum'])
            sp_delta.append(spy_df.loc[i + 1, 'Slow Pressure'] - spy_df.loc[i, 'Slow Pressure'])
            fp_delta.append(spy_df.loc[i + 1, 'Fast Pressure'] - spy_df.loc[i, 'Fast Pressure'])

    spy_df['m_delta'] = m_delta
    spy_df['sp_delta'] = sp_delta
    spy_df['fp_delta'] = fp_delta

    spy_df.drop(index=spy_df.index[0], axis=0, inplace=True)
    #spy_df.reset_index(inplace=True)

    matched_index = spy_df.loc[spy_df['time'] == last_entry['time']].index
    integer_index = matched_index[0]

    return spy_df.loc[integer_index-21:, :]

def add_new_data_to_database(spy_db, spy_df):
    # Add the rows to the DB
    for index, row in spy_df.iterrows():
        spy_db.put({'time': row['time'], 'open': row['open'], 'high': row['high'], 'low': row['low'],
                    'close': row['close'], 'Momemtum': row['Momemtum'], 'Slow Pressure': row['Slow Pressure'],
                    'Fast Pressure': row['Fast Pressure'], 'm_delta': row['m_delta'],
                    'sp_delta': row['sp_delta'], 'fp_delta': row['fp_delta'],
                    'change_close_open': row['change_close_open'], 'change_high_open': row['change_high_open'],
                    'change_low_open': row['change_low_open']}, key=row['time'])
    return

def create_call_model(db_df, winInput, drawdownInput):
    wins_drawdown = []
    w_or_l = []
    next_close_change = []

    for i in range((len(db_df) - 1)):
        if db_df.loc[i + 1, 'change_close_open'] > winInput and db_df.loc[i + 1, 'change_low_open'] > drawdownInput:
            wins_drawdown.append(db_df.loc[i + 1, 'change_low_open'])
            w_or_l.append(1)
        else:
            w_or_l.append(0)
        next_close_change.append(db_df.loc[i + 1, 'change_close_open'])
    w_or_l.append(0)
    next_close_change.append(0)


    st.write("Average Drawdown on ", len(wins_drawdown), ": ", np.average(wins_drawdown))

    db_df['w_or_l'] = w_or_l
    db_df['next_change_close'] = next_close_change

    # X_feed = db_df[db_df['w_or_l'] >= 0]
    X = db_df.drop(['time', 'w_or_l', 'open', 'high', 'low', 'close', 'key', 'next_change_close'], axis=1)
    y = db_df['next_change_close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

  #  dt = DecisionTreeClassifier(max_depth=3, random_state=12)
  #  clf = KNeighborsClassifier(n_neighbors=3)
  #  dt.fit(X_train, y_train)
  #  clf.fit(X_train, y_train)
    # Import RandomForestRegressor


    # Instantiate rf
    rf = RandomForestRegressor(n_estimators=25,
                               random_state=2)

    # Fit rf to the training set
    rf.fit(X_train, y_train)

    # Predict the test set labels
    y_pred = rf.predict(X_test)

    # Evaluate the test set RMSE
    rmse_test = MSE(y_test, y_pred) ** (1 / 2)

    # Print rmse_test
    st.write('Test set RMSE of rf: ', rmse_test)

    # Create a pd.Series of features importances
    importances = pd.Series(data=rf.feature_importances_,
                            index=X_train.columns)

    # Sort importances
    importances_sorted = importances.sort_values()

    st.dataframe(importances_sorted)
    st.bar_chart(importances_sorted)





  #  y_pred = clf.predict(X_test)
 #   accy = accuracy_score(y_test, y_pred)

    i = 0
    count_wins = 0
    count_losses = 0
    for entry in y_pred:
        if entry == 1 and y_test[i] == 1:
            count_wins += 1
        if entry == 1 and y_test[i] == 0:
            count_losses += 1
        i += 1




  #  st.write("Accuracy: ", accy)
    st.write("Number of wins predicted: ", count_wins, "/", len(y_pred))
    st.write("Number of losses predicted: ", count_losses, "/", len(y_pred))

    results_df = pd.DataFrame({'pred': y_pred, 'actual': y_test})
    st.write(results_df)

    return rf #dt

def create_put_model(db_df, winInput, drawdownInput):
    wins_drawdown = []
    w_or_l = []

    for i in range((len(db_df) - 1)):
        # if db_df.loc[i, 'm_delta'] < momentumInput and db_df.loc[i, 'sp_delta'] < spInput and \
        #         db_df.loc[i, 'fp_delta'] < fpInput:
        if db_df.loc[i + 1, 'change_close_open'] < winInput: # and db_df.loc[i + 1, 'change_high_open'] < drawdownInput:
            wins_drawdown.append(db_df.loc[i + 1, 'change_high_open'])
            w_or_l.append(1)
        else:
            w_or_l.append(0)
    w_or_l.append(0)

    st.write("Average Drawdown on ", len(wins_drawdown), ": ", np.average(wins_drawdown))

    db_df['w_or_l'] = w_or_l

    # X_feed = db_df[db_df['w_or_l'] >= 0]
    X = db_df.drop(['time', 'w_or_l', 'open', 'high', 'low', 'close', 'key'], axis=1).values
    y = db_df['w_or_l'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)

    dt = DecisionTreeClassifier(max_depth=2, random_state=12)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accy = accuracy_score(y_test, y_pred)

    st.write("Accuracy: ", accy)

    results_df = pd.DataFrame({'pred': y_pred, 'actual': y_test})
    st.write(results_df)


    return dt

def backtester(db_df, dt, spy_models):
    wins = []
    loses = []
    download5 = spy_models.get('new_call_dt_model_50.pkl')
    new_dt5 = pickle.loads(download5.read())
    remove_list = ['time', 'open', 'high', 'low', 'close', 'key']

    for row in db_df:
        for key in remove_list:
            del row[key]
        predictor_df = pd.DataFrame(data=row, index=[0]).values
        dt.predict(predictor_df)


def main():
    st.set_page_config(
        page_title="Stonkz Predictor",
        page_icon="🤑",
        layout="wide",)

    with st.spinner('Connecting Database'):
        #deta, spy_db, spy_models = connect_database()
       # connect_database()
        newDataTab, modelsTab, predictorTab = st.tabs(['Upload Data', 'Train and Save Models', 'Predictions'])


        # deta = Deta(st.secrets['DB_TOKEN'])
        # spy_db = deta.Base('spy_db')
        # spy_models = deta.Drive('spy_models')

        with newDataTab:
            # Upload a csv export file
            fileUpload = st.file_uploader('Upload SPY 1 HR Chart Data', type='csv')

            if st.button('Create DB Backup'):
                res = st.session_state.spy_db.fetch()
                allItems = res.items

                while res.last:
                    res = st.session_state.spy_db.fetch(last=res.last)
                    allItems += res.items

                db_df = pd.DataFrame(allItems)

                csv = db_df.to_csv().encode('utf-8')

                st.download_button(
                    label='Download DB Backup',
                    data=csv,
                    file_name=datetime.date.today().strftime('%m-%d-%Y')+'.csv',
                    mime='text/csv',
                )

            if fileUpload is not None:
                newData_df = pd.read_csv(fileUpload)

                spy_df = process_data(st.session_state.spy_db, newData_df)
                # # trim the upload to just columns we care about
                # spy_df = newData_df[
                #     ['time', 'open', 'high', 'low', 'close', 'Momemtum', 'Slow Pressure', 'Fast Pressure']].copy()
                #
                # spy_df['change_close_open'] = spy_df['close'] - spy_df['open']
                # spy_df['change_high_open'] = spy_df['high'] - spy_df['open']
                # spy_df['change_low_open'] = spy_df['low'] - spy_df['open']
                #
                # # add column for the deltas for momentum, sp, fp
                # m_delta = [0]
                # sp_delta = [0]
                # fp_delta = [0]
                #
                # for i in range(len(spy_df['Momemtum'])):
                #
                #     if i < len(spy_df['Momemtum']) - 1:
                #         m_delta.append(spy_df.loc[i + 1, 'Momemtum'] - spy_df.loc[i, 'Momemtum'])
                #         sp_delta.append(spy_df.loc[i + 1, 'Slow Pressure'] - spy_df.loc[i, 'Slow Pressure'])
                #         fp_delta.append(spy_df.loc[i + 1, 'Fast Pressure'] - spy_df.loc[i, 'Fast Pressure'])
                #
                # spy_df['m_delta'] = m_delta
                # spy_df['sp_delta'] = sp_delta
                # spy_df['fp_delta'] = fp_delta
                #
                # spy_df.drop(index=spy_df.index[0], axis=0, inplace=True)

                add_new_data_to_database(st.session_state.spy_db, spy_df)
                # # Add the rows to the DB
                # for index, row in spy_df.iterrows():
                #     spy_db.put({'time': row['time'], 'open': row['open'], 'high': row['high'], 'low': row['low'],
                #                 'close': row['close'], 'Momemtum': row['Momemtum'], 'Slow Pressure': row['Slow Pressure'],
                #                 'Fast Pressure': row['Fast Pressure'], 'm_delta': row['m_delta'],
                #                 'sp_delta': row['sp_delta'], 'fp_delta': row['fp_delta'],
                #                 'change_close_open': row['change_close_open'], 'change_high_open': row['change_high_open'],
                #                 'change_low_open': row['change_low_open']}, key=row['time'])

        with modelsTab:

            calls_or_puts = st.radio(
                'Choose to build a model for calls or puts',
                key='calls_or_puts',
                options=['Calls', 'Puts'],
            )

            winInput = st.slider('Choose a Win Threshold', -1.0, 1.0, step=0.1, value=0.5)
            drawdownInput = st.slider('Choose a Drawdown Threshold', -0.5, 0.5, step=0.1, value=-0.4)

            if st.button('Generate Model'):
                res = st.session_state.spy_db.fetch()
                allItems = res.items

                while res.last:
                    res = st.session_state.spy_db.fetch(last=res.last)
                    allItems += res.items

                db_df = pd.DataFrame(allItems)


                if calls_or_puts == 'Calls':

                    dt = create_call_model(db_df, winInput, drawdownInput)

                    st.download_button(
                        "Download Model",
                        data=pickle.dumps(dt),
                        file_name="dt_model.pkl",
                    )

                if calls_or_puts == 'Puts':

                    dt = create_put_model(db_df, winInput, drawdownInput)

                    st.download_button(
                        "Download Model",
                        data=pickle.dumps(dt),
                        file_name="dt_model.pkl",
                    )

        with predictorTab:

            CallTab, PutTab = st.tabs(['Calls', 'Puts'])

            with CallTab:


                c_download = st.session_state.spy_models.get('call_dt_model.pkl')
                c_download2 = st.session_state.spy_models.get('dt1_model.pkl')
                c_download3 = st.session_state.spy_models.get('rf1_model.pkl')
                c_new_dt = pickle.loads(c_download.read())
                c_new_dt2 = pickle.loads(c_download2.read())
                c_new_rf3 = pickle.loads(c_download3.read())

                pred_date = st.date_input('Choose Date', datetime.date.today(), key='call_date')
                #pred_time = st.selectbox('Choose Candle', ['06:30', '07:30', '08:30', '09:30', '10:30', '11:30', '12:30'], key='call_time_selction')
                pred_time = ['06:30', '07:30', '08:30', '09:30', '10:30', '11:30', '12:30']

                for time in pred_time:
                    if time == '06:30':
                        next_time = '07:30'
                    elif time == '07:30':
                        next_time = '08:30'
                    elif time == '08:30':
                        next_time = '09:30'
                    elif time == '09:30':
                        next_time = '10:30'
                    elif time == '10:30':
                        next_time = '11:30'
                    elif time == '11:30':
                        next_time = '12:30'
                    else:
                        next_time = '12:30'

                    candle_string = str(pred_date) + 'T' + str(time) + ':00-08:00'
                    next_candle_string = str(pred_date) + 'T' + str(next_time) + ':00-08:00'
                    selected_candle_data = st.session_state.spy_db.get(candle_string)
                    next_selected_candle_data = st.session_state.spy_db.get(next_candle_string)
                    st.dataframe(data=(pd.DataFrame(selected_candle_data, index=[0])),hide_index=True)


                    if selected_candle_data is not None:
                    #     download = spy_models.get('call_dt_model.pkl')
                    #     download2 = spy_models.get('dt1_model.pkl')
                    #     download3 = spy_models.get('rf1_model.pkl')
                    #     new_dt = pickle.loads(download.read())
                    #     new_dt2 = pickle.loads(download2.read())
                    #     new_rf3 = pickle.loads(download3.read())


                        close_price = selected_candle_data['close']
                        remove_list = ['time', 'open', 'high', 'low', 'close', 'key']
                        for key in remove_list:
                            del selected_candle_data[key]

                        predictor_df = pd.DataFrame(data=selected_candle_data, index=[0]).values
                        #st.write(predictor_df)

                        if c_new_dt.predict(predictor_df) == 1:
                            st.write("ML Says Buy", " - ", close_price, " Target: ", close_price+0.5)
                            if next_selected_candle_data['change_high_open'] > 0.4:
                                st.write("Win: High - ", next_selected_candle_data['high'], round(next_selected_candle_data['change_high_open'],2), " Low - ", next_selected_candle_data['low'], round(next_selected_candle_data['change_low_open'],2))
                            else:
                                st.write("Loss: High - ", next_selected_candle_data['high'], round(next_selected_candle_data['change_high_open'],2), " Low - ", next_selected_candle_data['low'], round(next_selected_candle_data['change_low_open'],2))
                        else:
                            st.write("ML Says Wait")

                        if c_new_dt2.predict(predictor_df) == 1:
                            st.write("ML NEW 50 Says Buy")
                            pred_price = c_new_rf3.predict(predictor_df)
                            st.write("Predicted Price: ", round((close_price+float(pred_price)),2))
                        else:
                            st.write("ML NEW 50 Says Wait")




            with PutTab:

                p_download = st.session_state.spy_models.get('put_dt_model.pkl')
                p_download2 = st.session_state.spy_models.get('put_dt1_model.pkl')
                p_download3 = st.session_state.spy_models.get('put_rf1_model.pkl')
                p_new_dt = pickle.loads(p_download.read())
                p_new_dt2 = pickle.loads(p_download2.read())
                p_new_rf3 = pickle.loads(p_download3.read())

                pred_date = st.date_input('Choose Date', datetime.date.today(), key='put_date')
                #pred_time = st.selectbox('Choose Candle', ['06:30', '07:30', '08:30', '09:30', '10:30', '11:30', '12:30'], key='put_time_selection')
                pred_time = ['06:30', '07:30', '08:30', '09:30', '10:30', '11:30', '12:30']

                for time in pred_time:
                    if time == '06:30':
                        next_time = '07:30'
                    elif time == '07:30':
                        next_time = '08:30'
                    elif time == '08:30':
                        next_time = '09:30'
                    elif time == '09:30':
                        next_time = '10:30'
                    elif time == '10:30':
                        next_time = '11:30'
                    elif time == '11:30':
                        next_time = '12:30'
                    else:
                        next_time = '12:30'

                    candle_string = str(pred_date) + 'T' + str(time) + ':00-08:00'
                    next_candle_string = str(pred_date) + 'T' + str(next_time) + ':00-08:00'
                    selected_candle_data = st.session_state.spy_db.get(candle_string)
                    next_selected_candle_data = st.session_state.spy_db.get(next_candle_string)
                    st.dataframe(data=(pd.DataFrame(selected_candle_data, index=[0])),hide_index=True)

                    if selected_candle_data is not None:
                    #     download = spy_models.get('put_dt_model.pkl')
                    #     download2 = spy_models.get('put_dt1_model.pkl')
                    #     download3 = spy_models.get('put_rf1_model.pkl')
                    #     new_dt = pickle.loads(download.read())
                    #     new_dt2 = pickle.loads(download2.read())
                    #     new_rf3 = pickle.loads(download3.read())

                        close_price = selected_candle_data['close']
                        remove_list = ['time', 'open', 'high', 'low', 'close', 'key']
                        for key in remove_list:
                            del selected_candle_data[key]

                        predictor_df = pd.DataFrame(data=selected_candle_data, index=[0]).values
                        #st.write(predictor_df)

                        if p_new_dt.predict(predictor_df) == 1:
                            st.write("ML Says Buy", " - ", close_price, " Target: ", close_price-0.5)
                            if (next_selected_candle_data is not None):
                                if next_selected_candle_data['change_low_open'] < -0.5:
                                    st.write("Win: Low - ", next_selected_candle_data['low'], round(next_selected_candle_data['change_low_open'],2), " High - ", next_selected_candle_data['high'], round(next_selected_candle_data['change_high_open'],2))
                                else:
                                    st.write("Loss: Low - ", next_selected_candle_data['low'], round(next_selected_candle_data['change_low_open'],2), " High - ", next_selected_candle_data['high'], round(next_selected_candle_data['change_high_open'],2))
                        else:
                            st.write("ML Says Wait")

                        if p_new_dt2.predict(predictor_df) == 1:
                            st.write("ML NEW 50 Says Buy")
                            pred_price = p_new_rf3.predict(predictor_df)
                            st.write("Predicted Price: ", round((close_price+float(pred_price)), 2))
                        else:
                            st.write("ML NEW 50 Says Wait")


    return


if __name__ == "__main__":
    main()
