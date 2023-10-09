import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from deta import Deta
import pickle
import datetime


def connect_database():
    deta = Deta(st.secrets['DB_TOKEN'])
    spy_db = deta.Base('spy_db')
    spy_models = deta.Drive('spy_models')

    return deta, spy_db, spy_models

def process_data(spy_db, newData_df):
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

    return spy_df

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
    w_or_l = [0]

    for i in range((len(db_df) - 1)):
        # if db_df.loc[i, 'm_delta'] > momentumInput and db_df.loc[i, 'sp_delta'] > spInput and db_df.loc[
        #     i, 'fp_delta'] > fpInput:
        # st.write(i+1, db_df.loc[i+1, 'change'])
        if db_df.loc[i + 1, 'change_close_open'] > winInput and db_df.loc[i + 1, 'change_low_open'] > drawdownInput:
            wins_drawdown.append(db_df.loc[i + 1, 'change_low_open'])
            w_or_l.append(1)
        else:
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

    return dt

def create_put_model(db_df, winInput, drawdownInput):
    wins_drawdown = []
    w_or_l = [0]

    for i in range((len(db_df) - 1)):
        # if db_df.loc[i, 'm_delta'] < momentumInput and db_df.loc[i, 'sp_delta'] < spInput and \
        #         db_df.loc[i, 'fp_delta'] < fpInput:
        if db_df.loc[i + 1, 'change_close_open'] < winInput and db_df.loc[i + 1, 'change_high_open'] < drawdownInput:
            wins_drawdown.append(db_df.loc[i + 1, 'change_high_open'])
            w_or_l.append(1)
        else:
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

    return dt

def main():
    st.set_page_config(
        page_title="Stonkz Predictor",
        page_icon="🤑",
        layout="wide",)

    newDataTab, modelsTab, predictorTab = st.tabs(['Upload Data', 'Train and Save Models', 'Predictions'])

    deta, spy_db, spy_models = connect_database()
    # deta = Deta(st.secrets['DB_TOKEN'])
    # spy_db = deta.Base('spy_db')
    # spy_models = deta.Drive('spy_models')

    with newDataTab:
        # Upload a csv export file
        fileUpload = st.file_uploader('Upload SPY 1 HR Chart Data', type='csv')

        if fileUpload is not None:
            newData_df = pd.read_csv(fileUpload)

            spy_df = process_data(spy_db, newData_df)
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

            add_new_data_to_database(spy_db, spy_df)
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
            res = spy_db.fetch()
            allItems = res.items

            while res.last:
                res = spy_db.fetch(last=res.last)
                allItems += res.items

            db_df = pd.DataFrame(allItems)


            if calls_or_puts == 'Calls':

                # wins_drawdown = []
                # w_or_l = [0]
                #
                # for i in range((len(db_df) - 1)):
                #     # if db_df.loc[i, 'm_delta'] > momentumInput and db_df.loc[i, 'sp_delta'] > spInput and db_df.loc[
                #     #     i, 'fp_delta'] > fpInput:
                #         # st.write(i+1, db_df.loc[i+1, 'change'])
                #     if db_df.loc[i + 1, 'change_close_open'] > winInput and db_df.loc[i + 1, 'change_low_open'] > drawdownInput:
                #         wins_drawdown.append(db_df.loc[i + 1, 'change_low_open'])
                #         w_or_l.append(1)
                #     else:
                #         w_or_l.append(0)
                #
                # st.write("Average Drawdown on ", len(wins_drawdown), ": ", np.average(wins_drawdown))
                #
                # db_df['w_or_l'] = w_or_l
                #
                #
                # #X_feed = db_df[db_df['w_or_l'] >= 0]
                # X = db_df.drop(['time', 'w_or_l', 'open', 'high', 'low', 'close', 'key'], axis=1).values
                # y = db_df['w_or_l'].values
                #
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)
                #
                # dt = DecisionTreeClassifier(max_depth=2, random_state=12)
                # dt.fit(X_train, y_train)
                #
                # y_pred = dt.predict(X_test)
                # accy = accuracy_score(y_test, y_pred)
                #
                # st.write("Accuracy: ", accy)
                #
                # results_df = pd.DataFrame({'pred': y_pred, 'actual': y_test})

                dt = create_call_model(db_df, winInput, drawdownInput)

                st.download_button(
                    "Download Model",
                    data=pickle.dumps(dt),
                    file_name="dt_model.pkl",
                )

            if calls_or_puts == 'Puts':

                # wins_drawdown = []
                # w_or_l = [0]
                #
                # for i in range((len(db_df)-1)):
                #      # if db_df.loc[i, 'm_delta'] < momentumInput and db_df.loc[i, 'sp_delta'] < spInput and \
                #      #         db_df.loc[i, 'fp_delta'] < fpInput:
                #      if db_df.loc[i + 1, 'change_close_open'] < winInput and db_df.loc[i + 1, 'change_high_open'] < drawdownInput:
                #          wins_drawdown.append(db_df.loc[i + 1, 'change_high_open'])
                #          w_or_l.append(1)
                #      else:
                #          w_or_l.append(0)
                #      # else:
                #      #     both.append(0)
                #
                # st.write("Average Drawdown on ", len(wins_drawdown), ": ", np.average(wins_drawdown))
                #
                # db_df['w_or_l'] = w_or_l
                #
                #
                # #X_feed = db_df[db_df['w_or_l'] >= 0]
                # X = db_df.drop(['time', 'w_or_l', 'open', 'high', 'low', 'close', 'key'], axis=1).values
                # y = db_df['w_or_l'].values
                #
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)
                #
                # dt = DecisionTreeClassifier(max_depth=2, random_state=12)
                # dt.fit(X_train, y_train)
                #
                # y_pred = dt.predict(X_test)
                # accy = accuracy_score(y_test, y_pred)
                #
                # st.write("Accuracy: ", accy)
                #
                # results_df = pd.DataFrame({'pred': y_pred, 'actual': y_test})
                dt = create_put_model(db_df, winInput, drawdownInput)

                st.download_button(
                    "Download Model",
                    data=pickle.dumps(dt),
                    file_name="dt_model.pkl",
                )

            # with open('dt_model.pkl', 'wb') as f:
            #   pickle.dump(dt, f)

            #####spy_models.put('dt_model.pkl', f)

    with predictorTab:

        CallTab, PutTab = st.tabs(['Calls', 'Puts'])

        with CallTab:

           # res = spy_db.fetch()
           # allItems = res.items

         #   while res.last:
         #       res = spy_db.fetch(last=res.last)
          #      allItems += res.items

        #    db_df = pd.DataFrame(allItems)

            pred_date = st.date_input('Choose Date', datetime.date.today(), key='call_date')
            pred_time = st.time_input('Choose Candle', datetime.time(7,30), step=1800, key='call_time')
            candle_string = str(pred_date) + 'T' + str(pred_time) + '-07:00'
            #td = datetime.timedelta(hours=1)
            #candle_string_prev = str(pred_date) + 'T' + str(pred_time-td) + '-07:00'
            selected_candle_data = spy_db.get(candle_string)
            st.write(selected_candle_data)

            if selected_candle_data is not None:
                download = spy_models.get('call_dt_model.pkl')
                download2 = spy_models.get('call_dt_model_70_30.pkl')
                download3 = spy_models.get('call_dt_model_100_20.pkl')
                download4 = spy_models.get('call_dt_model_50_10.pkl')
                new_dt = pickle.loads(download.read())
                new_dt2 = pickle.loads(download2.read())
                new_dt3 = pickle.loads(download3.read())
                new_dt4 = pickle.loads(download4.read())

                remove_list = ['time', 'open', 'high', 'low', 'close', 'key']
                for key in remove_list:
                    del selected_candle_data[key]

                predictor_df = pd.DataFrame(data=selected_candle_data, index=[0]).values
                st.write(predictor_df)

                if new_dt.predict(predictor_df) == 1:
                    st.write("ML Says Buy")
                else:
                    st.write("ML Says Wait")

                if new_dt2.predict(predictor_df) == 1:
                    st.write("ML 70/30 Says Buy")
                else:
                    st.write("ML 70/30 Says Wait")

                if new_dt3.predict(predictor_df) == 1:
                    st.write("ML 100/20 Says Buy")
                else:
                    st.write("ML 100/20 Says Wait")

                if new_dt4.predict(predictor_df) == 1:
                    st.write("ML 50/10 Says Buy")
                else:
                    st.write("ML 50/10 Says Wait")


        with PutTab:

            # calcValue = st.radio(
            #     'Choose to use High or Close for Calc',
            #     key='put_calc_value',
            #     options=['high', 'low', 'close'],
            #     index=2,
            # )

            # res = spy_db.fetch()
            # allItems = res.items
            #
            # while res.last:
            #     res = spy_db.fetch(last=res.last)
            #     allItems += res.items
            #
            # db_df = pd.DataFrame(allItems)
            #
            # db_df['change'] = db_df[calcValue] - db_df['open']
            #
            # # add column for the deltas for momentum, sp, fp
            # m_delta = [0]
            # sp_delta = [0]
            # fp_delta = [0]
            #
            # for i in range(len(db_df['Momemtum'])):
            #
            #     if i < len(db_df['Momemtum']) - 1:
            #         m_delta.append(db_df.loc[i + 1, 'Momemtum'] - db_df.loc[i, 'Momemtum'])
            #         sp_delta.append(db_df.loc[i + 1, 'Slow Pressure'] - db_df.loc[i, 'Slow Pressure'])
            #         fp_delta.append(db_df.loc[i + 1, 'Fast Pressure'] - db_df.loc[i, 'Fast Pressure'])
            #
            # db_df['m_delta'] = m_delta
            # db_df['sp_delta'] = sp_delta
            # db_df['fp_delta'] = fp_delta
            #
            #
            # download = spy_models.get('dt_model.pkl')
            # new_dt = pickle.loads(download.read())
            #
            # predictor_df = pd.DataFrame(
            #     db_df.iloc[-2].drop(['time', 'open', 'high', 'low', 'close', 'key'])).values
            # st.write(predictor_df)
            # if new_dt.predict(predictor_df.T) == 1:
            #     st.write("ML Says Buy")
            # else:
            #     st.write("ML Says Wait")

            pred_date = st.date_input('Choose Date', datetime.date.today(), key='put_date')
            pred_time = st.time_input('Choose Candle', datetime.time(7, 30), step=1800, key='put_time')
            candle_string = str(pred_date) + 'T' + str(pred_time) + '-07:00'
            # td = datetime.timedelta(hours=1)
            # candle_string_prev = str(pred_date) + 'T' + str(pred_time-td) + '-07:00'
            st.write(candle_string)
            selected_candle_data = spy_db.get(candle_string)
            st.write(selected_candle_data)

            if selected_candle_data is not None:
                download = spy_models.get('put_dt_model.pkl')
                download2 = spy_models.get('put_dt_model_70_30.pkl')
                download3 = spy_models.get('put_dt_model_100_20.pkl')
                download4 = spy_models.get('put_dt_model_50_10.pkl')
                new_dt = pickle.loads(download.read())
                new_dt2 = pickle.loads(download2.read())
                new_dt3 = pickle.loads(download3.read())
                new_dt4 = pickle.loads(download4.read())

                remove_list = ['time', 'open', 'high', 'low', 'close', 'key']
                for key in remove_list:
                    del selected_candle_data[key]

                predictor_df = pd.DataFrame(data=selected_candle_data, index=[0]).values
                st.write(predictor_df)

                if new_dt.predict(predictor_df) == 1:
                    st.write("ML Says Buy")
                else:
                    st.write("ML Says Wait")

                if new_dt2.predict(predictor_df) == 1:
                    st.write("ML 70/30 Says Buy")
                else:
                    st.write("ML 70/30 Says Wait")

                if new_dt3.predict(predictor_df) == 1:
                    st.write("ML 100/20 Says Buy")
                else:
                    st.write("ML 100/20 Says Wait")

                if new_dt4.predict(predictor_df) == 1:
                    st.write("ML 50/10 Says Buy")
                else:
                    st.write("ML 50/10 Says Wait")


    return


if __name__ == "__main__":
    main()
