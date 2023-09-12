import pandas as pd
import numpy as np
import streamlit as st
def main():
    # Upload a csv export file
    fileUpload = st.file_uploader('Choose your file', type='csv')
    if fileUpload is not None:
        master_df = pd.read_csv(fileUpload)
        trimmed_df = master_df[
            ['time', 'open', 'high', 'low', 'close', 'Momemtum', 'Slow Pressure', 'Fast Pressure']].copy()

        # Choose to use the high or the close for the calculation of change
        calcValue = st.radio(
            'Choose to use High or Close for Calc',
            key='calc_value',
            options=['high', 'low', 'close'],
        )

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
            both = []

            momentumInput = st.slider('Choose Momentum Threshold', 0, 30, value=10)
            spInput = st.slider('Choose Slow Pressure Threshold', 0, 50, value=0)
            fpInput = st.slider('Choose Fast Pressure Threshold', 0, 200, value=50)
            winInput = st.slider('Choose a Win Threshold', 0.0, 1.0, step=0.1, value=0.5)

            for i in range(len(trimmed_df)):
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
                    both.append(0)

            trimmed_df['w_or_l'] = both

            st.write('CountWin: ', count_win, np.mean(wins))
            st.write('CountLose:', count_lose, np.mean(loses))

            lastRow = trimmed_df.tail(1)
            prediction = lastRow['w_or_l'].iloc[0]

            st.write(lastRow)

            if prediction == 1:
                st.write('BUY!')
            else:
                st.write('WAIT!')

        with PutTab:
            # Count the wins/loses
            count_win = 0
            count_lose = 0
            wins = []
            loses = []
            both = []

            momentumInput = st.slider('Choose Momentum Threshold', -30, 0, value=-10)
            spInput = st.slider('Choose Slow Pressure Threshold', -50, 0, value=0)
            fpInput = st.slider('Choose Fast Pressure Threshold', -200, 0, value=-50)
            winInput = st.slider('Choose a Win Threshold', -1.0, 0.0, step=0.1, value=-0.5)

            for i in range(len(trimmed_df)):
                if trimmed_df.loc[i, 'm_delta'] < momentumInput and trimmed_df.loc[i, 'sp_delta'] < spInput and \
                        trimmed_df.loc[i, 'fp_delta'] < fpInput:
                    # st.write(i+1, trimmed_df.loc[i+1, 'change'])
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

            lastRow = trimmed_df.tail(1)
            prediction = lastRow['w_or_l'].iloc[0]

            st.write(lastRow)

            if prediction == 1:
                st.write('BUY!')
            else:
                st.write('WAIT!')


        return


if __name__ == "__main__":
    main()
