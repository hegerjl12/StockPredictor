import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from deta import Deta
import pickle
import datetime


def main():
    st.set_page_config(
        page_title="Stonkz Predictor",
        page_icon="🤑",
        layout="wide",)

    deta = Deta(st.secrets['DB_TOKEN'])
    spy_db = deta.Base('spy_db')
    spy_models = deta.Drive('spy_models')

    newDataTab, modelsTab, predictorTab = st.tabs(['Upload Data', 'Train and Save Models', 'Predictions'])

    with newDataTab:
        # Upload a csv export file
        fileUpload = st.file_uploader('Upload SPY 1 HR Chart Data', type='csv')

        if fileUpload is not None:
            newData_df = pd.read_csv(fileUpload)

            # trim the upload to just columns we care about
            spy_df = newData_df[
                ['time', 'open', 'high', 'low', 'close', 'Momemtum', 'Slow Pressure', 'Fast Pressure']].copy()

            # Add the rows to the DB
            for index, row in spy_df.iterrows():
                spy_db.put({'time': row['time'], 'open': row['open'], 'high': row['high'], 'low': row['low'],
                            'close': row['close'], 'Momemtum': row['Momemtum'], 'Slow Pressure': row['Slow Pressure'],
                            'Fast Pressure': row['Fast Pressure']}, key=row['time'])

    with modelsTab:
        # Choose to use the high or the close for the calculation of change
        calcValue = st.radio(
            'Choose to use High or Close for Calc',
            key='calc_value',
            options=['high', 'low', 'close'],
            index=2,
        )

        calls_or_puts = st.radio(
            'Choose to build a model for calls or puts',
            key='calls_or_puts',
            options=['Calls', 'Puts'],
        )

        momentumInput = st.slider('Choose Momentum Threshold', -30, 30, value=0)
        spInput = st.slider('Choose Slow Pressure Threshold', -50, 50, value=0)
        fpInput = st.slider('Choose Fast Pressure Threshold', -200, 200, value=0)
        winInput = st.slider('Choose a Win Threshold', 0.0, 1.0, step=0.1, value=0.5)

        if st.button('Generate Model'):
            res = spy_db.fetch()
            allItems = res.items

            while res.last:
                res = spy_db.fetch(last=res.last)
                allItems += res.items

            db_df = pd.DataFrame(allItems)

            db_df['change'] = db_df[calcValue] - db_df['open']

            # add column for the deltas for momentum, sp, fp
            m_delta = [0]
            sp_delta = [0]
            fp_delta = [0]

            for i in range(len(db_df['Momemtum'])):

                if i < len(db_df['Momemtum']) - 1:
                    m_delta.append(db_df.loc[i + 1, 'Momemtum'] - db_df.loc[i, 'Momemtum'])
                    sp_delta.append(db_df.loc[i + 1, 'Slow Pressure'] - db_df.loc[i, 'Slow Pressure'])
                    fp_delta.append(db_df.loc[i + 1, 'Fast Pressure'] - db_df.loc[i, 'Fast Pressure'])

            db_df['m_delta'] = m_delta
            db_df['sp_delta'] = sp_delta
            db_df['fp_delta'] = fp_delta

            if calls_or_puts == 'Calls':
                # Count the wins/loses
                count_win = 0
                count_lose = 0
                wins = []
                loses = []
                both = [0]

                for i in range((len(db_df) - 1)):
                    if db_df.loc[i, 'm_delta'] > momentumInput and db_df.loc[i, 'sp_delta'] > spInput and db_df.loc[
                        i, 'fp_delta'] > fpInput:
                        # st.write(i+1, db_df.loc[i+1, 'change'])
                        if db_df.loc[i + 1, 'change'] > winInput:
                            count_win += 1
                            wins.append(db_df.loc[i + 1, 'change'])
                            both.append(1)
                        else:
                            count_lose += 1
                            loses.append(db_df.loc[i + 1, 'change'])
                            both.append(0)
                    else:
                        both.append(-1)

                db_df['w_or_l'] = both

                if count_win + count_lose > 0:
                    winPercentage = count_win / (count_win + count_lose) * 100
                else:
                    winPercentage = 1

                st.write('CountWin: ', count_win, np.mean(wins))
                st.write('CountLose:', count_lose, np.mean(loses))
                st.write('Win Percent: ', winPercentage, '%')

                X_feed = db_df[db_df['w_or_l'] >= 0]
                X = X_feed.drop(['time', 'w_or_l', 'open', 'high', 'low', 'close', 'key'], axis=1).values
                y = X_feed['w_or_l'].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)

                dt = DecisionTreeClassifier(max_depth=2, random_state=12)
                dt.fit(X_train, y_train)

                y_pred = dt.predict(X_test)
                accy = accuracy_score(y_test, y_pred)

                st.write("Accuracy: ", accy)

                results_df = pd.DataFrame({'pred': y_pred, 'actual': y_test})

                st.download_button(
                    "Download Model",
                    data=pickle.dumps(dt),
                    file_name="dt_model.pkl",
                )
            if calls_or_puts == 'Puts':
                # Count the wins/loses
                count_win = 0
                count_lose = 0
                wins = []
                loses = []
                both = [0]

                for i in range((len(db_df)-1)):
                     if db_df.loc[i, 'm_delta'] < momentumInput and db_df.loc[i, 'sp_delta'] < spInput and \
                             db_df.loc[i, 'fp_delta'] < fpInput:
                         if db_df.loc[i + 1, 'change'] < winInput:
                             count_win += 1
                             wins.append(db_df.loc[i + 1, 'change'])
                             both.append(1)
                         else:
                             count_lose += 1
                             loses.append(db_df.loc[i + 1, 'change'])
                             both.append(0)
                     else:
                         both.append(0)

                db_df['w_or_l'] = both

                if count_win + count_lose > 0:
                    winPercentage = count_win / (count_win + count_lose)*100
                else:
                    winPercentage = 1

                st.write('CountWin: ', count_win, np.mean(wins))
                st.write('CountLose:', count_lose, np.mean(loses))
                st.write('Win Percent: ', winPercentage, '%')

                X_feed = db_df[db_df['w_or_l'] >= 0]
                X = X_feed.drop(['time', 'w_or_l', 'open', 'high', 'low', 'close', 'key'], axis=1).values
                y = X_feed['w_or_l'].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)

                dt = DecisionTreeClassifier(max_depth=2, random_state=12)
                dt.fit(X_train, y_train)

                y_pred = dt.predict(X_test)
                accy = accuracy_score(y_test, y_pred)

                st.write("Accuracy: ", accy)

                results_df = pd.DataFrame({'pred': y_pred, 'actual': y_test})

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
            calcValue = st.radio(
                'Choose to use High or Close for Calc',
                key='call_calc_value',
                options=['high', 'low', 'close'],
                index=2,
            )

           # res = spy_db.fetch()
           # allItems = res.items

         #   while res.last:
         #       res = spy_db.fetch(last=res.last)
          #      allItems += res.items

        #    db_df = pd.DataFrame(allItems)

            pred_date = st.date_input('Choose Date', datetime.date.today())
            pred_time = st.time_input('Choose Candle', datetime.time(6,30), step=1800)
            candle_string = str(pred_date) + 'T' + str(pred_time) + '-07:00'
            #candle_string_prev = pred_date + 'T' +
            st.write(candle_string)
            db_df = spy_db.get(candle_string)
            st.write(db_df)
            #db_df['change'] = db_df[calcValue] - db_df['open']

            # add column for the deltas for momentum, sp, fp
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
            # predictor_df = pd.DataFrame(db_df.iloc[-2].drop(['time', 'open', 'high', 'low', 'close', 'key'])).values
            # st.write(predictor_df)
            # if new_dt.predict(predictor_df.T) == 1:
            #     st.write("ML Says Buy")
            # else:
            #     st.write("ML Says Wait")


        with PutTab:
            st.write("placeholder")
            # calcValue = st.radio(
            #     'Choose to use High or Close for Calc',
            #     key='put_calc_value',
            #     options=['high', 'low', 'close'],
            #     index=2,
            # )
            #
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



    return


if __name__ == "__main__":
    main()
