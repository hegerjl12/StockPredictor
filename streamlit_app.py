import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from deta import Deta

def main():
    st.set_page_config(
        page_title="Stonkz Predictor",
        page_icon="🤑",
        layout="wide",)

    deta = Deta('b0hw2ref5az_HYnGn4FqS9gNhcBbT83D6KTVFsM52Hzz')
    spy_db = deta.Base('spy_db')


    # Upload a csv export file
    fileUpload = st.file_uploader('Choose your file', type='csv')
    if fileUpload is not None:
        master_df = pd.read_csv(fileUpload)
        column_keys = master_df.columns


        trimmed_df = master_df[
            ['time', 'open', 'high', 'low', 'close', 'Momemtum', 'Slow Pressure', 'Fast Pressure']].copy()

       # for index, row in trimmed_df.iterrows():
        #    spy_db.put({'time':row['time'], 'open':row['open'], 'high':row['high'], 'low':row['low'], 'close':row['close'], 'Momemtum':row['Momemtum'], 'Slow Pressure':row['Slow Pressure'], 'Fast Pressure':row['Fast Pressure']}, key=row['time'])

        # Choose to use the high or the close for the calculation of change
        calcValue = st.radio(
            'Choose to use High or Close for Calc',
            key='calc_value',
            options=['high', 'low', 'close'],
        )

        res = spy_db.fetch()
        allItems = res.items

        while res.last:
            res = spy_db.fetch(last=res.last)
            allItems += res.items

        st.write(allItems)

        trimmed_df['change'] = trimmed_df[calcValue] - trimmed_df['open']

        # add column for the deltas for momentum, sp, fp
        m_delta = [0]
        sp_delta = [0]
        fp_delta = [0]

        for i in range(len(trimmed_df['Momemtum'])):

            if i < len(trimmed_df['Momemtum']) - 1:
                m_delta.append(trimmed_df.loc[i + 1, 'Momemtum'] - trimmed_df.loc[i, 'Momemtum'])
                sp_delta.append(trimmed_df.loc[i + 1, 'Slow Pressure'] - trimmed_df.loc[i, 'Slow Pressure'])
                fp_delta.append(trimmed_df.loc[i + 1, 'Fast Pressure'] - trimmed_df.loc[i, 'Fast Pressure'])

        trimmed_df['m_delta'] = m_delta
        trimmed_df['sp_delta'] = sp_delta
        trimmed_df['fp_delta'] = fp_delta

        dfExpander = st.expander('Expand to see DF')
        dfExpander.dataframe(trimmed_df)

        CallTab, PutTab = st.tabs(['Calls', 'Puts'])

        with CallTab:
            # Count the wins/loses
            count_win = 0
            count_lose = 0
            wins = []
            loses = []
            both = [0]

            callInputExpander = st.expander('Expand to see inputs')
            with callInputExpander:

                momentumInput = st.slider('Choose Momentum Threshold', 0, 30, value=10)
                spInput = st.slider('Choose Slow Pressure Threshold', 0, 50, value=0)
                fpInput = st.slider('Choose Fast Pressure Threshold', 0, 200, value=50)
                winInput = st.slider('Choose a Win Threshold', 0.0, 1.0, step=0.1, value=0.5)

            for i in range((len(trimmed_df)-1)):
                if trimmed_df.loc[i, 'm_delta'] > momentumInput and trimmed_df.loc[i, 'sp_delta'] > spInput and trimmed_df.loc[i, 'fp_delta'] > fpInput:
                    #st.write(i+1, trimmed_df.loc[i+1, 'change'])
                    if trimmed_df.loc[i+1,'change'] > winInput:
                        count_win += 1
                        wins.append(trimmed_df.loc[i+1,'change'])
                        both.append(1)
                    else:
                        count_lose += 1
                        loses.append(trimmed_df.loc[i+1,'change'])
                        both.append(0)
                else:
                    both.append(-1)

            trimmed_df['w_or_l'] = both

            st.write('CountWin: ', count_win, np.mean(wins))
            st.write('CountLose:', count_lose, np.mean(loses))
            st.write('Win Percent: ', count_win/(count_win+count_lose)*100, '%')

            lastRow = trimmed_df.tail(1)

            st.write(lastRow)

            if lastRow['m_delta'].iloc[0] > momentumInput and lastRow['sp_delta'].iloc[0] > spInput and \
                    lastRow['fp_delta'].iloc[0] > fpInput:
                st.metric(label="BUY", value=lastRow['close'].iloc[0], delta="CALL")
            else:
                st.error('WAIT!')

            X_feed = trimmed_df[trimmed_df['w_or_l'] >= 0]
            X = X_feed.drop(['time', 'w_or_l', 'open', 'high', 'low', 'close'], axis=1).values
            y = X_feed['w_or_l'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)

            dt = DecisionTreeClassifier(max_depth=2, random_state=12)
            dt.fit(X_train, y_train)

            y_pred = dt.predict(X_test)
            accy = accuracy_score(y_test, y_pred)

            st.write("Accuracy: ", accy)

            results_df = pd.DataFrame({'pred': y_pred, 'actual':y_test})


            predictor_df = pd.DataFrame(trimmed_df.iloc[-2].drop(['time', 'open', 'high', 'low', 'close', 'w_or_l'])).values
            if dt.predict(predictor_df.T) == 1:
                st.write("ML Says Buy")
            else:
                st.write("ML Says Wait")

            st.write(results_df)


        with PutTab:
            # Count the wins/loses
            count_win = 0
            count_lose = 0
            wins = []
            loses = []
            both = [0]

            putInputExpander = st.expander('Expand to see inputs')
            with putInputExpander:

                momentumInput = st.slider('Choose Momentum Threshold', -30, 0, value=-10)
                spInput = st.slider('Choose Slow Pressure Threshold', -50, 0, value=0)
                fpInput = st.slider('Choose Fast Pressure Threshold', -200, 0, value=-50)
                winInput = st.slider('Choose a Win Threshold', -1.0, 0.0, step=0.1, value=-0.5)

            for i in range((len(trimmed_df)-1)):
                if trimmed_df.loc[i, 'm_delta'] < momentumInput and trimmed_df.loc[i, 'sp_delta'] < spInput and \
                        trimmed_df.loc[i, 'fp_delta'] < fpInput:

                    if trimmed_df.loc[i + 1, 'change'] < winInput:
                        count_win += 1
                        wins.append(trimmed_df.loc[i + 1, 'change'])
                        both.append(1)
                    else:
                        count_lose += 1
                        loses.append(trimmed_df.loc[i + 1, 'change'])
                        both.append(0)
                else:
                    both.append(0)

            trimmed_df['w_or_l'] = both

            st.write('CountWin: ', count_win, np.mean(wins))
            st.write('CountLose:', count_lose, np.mean(loses))
            st.write('Win Percent: ', count_win / (count_win + count_lose)*100, '%')

            lastRow = trimmed_df.tail(1)
            prediction = lastRow['w_or_l'].iloc[0]

            st.write(lastRow)

            if lastRow['m_delta'].iloc[0] < momentumInput and lastRow['sp_delta'].iloc[0] < spInput and \
                        lastRow['fp_delta'].iloc[0] < fpInput:
                st.metric(label="BUY", value=lastRow['close'].iloc[0], delta="-PUT")
            else:
                st.error('WAIT!')


        return


if __name__ == "__main__":
    main()
