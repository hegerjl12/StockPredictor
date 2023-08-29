import pandas as pd
import numpy as np
import streamlit as st


def main():
    filename = st.selectbox('Choose your file', ('test1', 'test15.csv', 'test30.csv', 'test60.csv', 'test120.csv', 'test1D.csv'))
    master_df = pd.read_csv(filename)
    trimmed_df = master_df[
        ['time', 'open', 'high', 'low', 'close', 'Momemtum', 'Slow Pressure', 'Fast Pressure']].copy()

    trimmed_df['change'] = trimmed_df['high'] - trimmed_df['open']

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

    st.dataframe(trimmed_df)

    count_win = 0
    count_lose = 0
    wins = []
    loses = []

    for i in range(len(trimmed_df)):
        if trimmed_df.loc[i, 'm_delta'] > 10 and trimmed_df.loc[i, 'sp_delta'] > 0 and trimmed_df.loc[i, 'fp_delta'] > 0:
            st.write(i+1, trimmed_df.loc[i+1, 'change'])
            if trimmed_df.loc[i+1,'change'] > 0:
                count_win += 1
                wins.append(trimmed_df.loc[i+1,'change'])
            else:
                count_lose += 1
                loses.append(trimmed_df.loc[i+1,'change'])

    st.write('CountWin: ', count_win, np.mean(wins))
    st.write('CountLose:', count_lose, np.mean(loses))

    return


if __name__ == "__main__":
    main()
