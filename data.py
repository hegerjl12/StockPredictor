import pandas as pd
import numpy as np
import streamlit as st
from deta import Deta
import datetime

def connect_to_database():
    deta = Deta(st.secrets['DB_TOKEN'])
    spy_db = deta.Base('spy_db')
    spy_models = deta.Drive('spy_models')

    return spy_db, spy_models
@st.cache
def backup_database(spy_db):
    csv = spy_db.to_csv().encode('utf-8')

    st.download_button(
        label='Download DB Backup',
        data=csv,
        file_name=datetime.date.today().strftime('%m-%d-%Y'),
        mime='text/csv',
    )

def calculate_new_data():

    return
def upload_new_data(spy_db, newData_df):

    return