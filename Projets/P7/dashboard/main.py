from data_viz import *
import streamlit as st
import aiohttp
import asyncio
from aiocache import Cache, cached
import pandas as pd
import dill
import codecs
import shap
# import traceback
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


dashboard_url = 'https://homecredit-api-oc.herokuapp.com/'
# dashboard_url = 'http://127.0.0.1:8000/'


def dillDecode(data):
    return dill.loads(codecs.decode(data.encode(), "base64"))


def dillEncode(data):
    return codecs.encode(dill.dumps(data), "base64").decode()


async def fetch(session, url):
    try:
        async with session.get(url) as response:
            result = await response.json()
            return result
    except Exception:
        return {}


async def post(session, url, data):
    try:
        async with session.post(url, params=data) as response:
            return await response.json()
    except Exception:
        # traceback.print_exc()
        return {}


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Appliquer filtres")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrer sur", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valeurs pour {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Valeurs pour {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valeurs pour {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(
                        map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Text dans {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


def filter_viz(df: pd.DataFrame):
    display = st.checkbox("Distributions")

    if display:
        modification_container = st.container()
        with modification_container:
            to_filter_columns = st.multiselect("Colonnes", df.columns)

            if len(to_filter_columns) >= 1:
                print(to_filter_columns)
                st.pyplot(histo_distribution(df[to_filter_columns]))


def compare_viz(df: pd.DataFrame):
    compare = st.checkbox("Comparaison")

    if compare:
        modification_container = st.container()
        with modification_container:
            to_filter_columns = st.multiselect("Colonnes", df.columns)
            if len(to_filter_columns) > 2:
                st.warning(
                    'Uniquement les deux premières colonnes sont prises en compte.', icon="⚠️")

            if len(to_filter_columns) >= 2:
                st.pyplot(dispersion(
                    df, to_filter_columns[0], to_filter_columns[1]))


@st.cache
def display_data(df):
    return filter_dataframe(df)


@cached(ttl=None, cache=Cache.MEMORY)
async def applicant_data(applicant):
    st.write('## Client Sélectionné')
    st.dataframe(applicant, width=1500)


@cached(ttl=None, cache=Cache.MEMORY)
async def prediction(applicant, nb_features_shap, _explainer, _model):
    applicant = applicant.drop(columns='SK_ID_CURR')
    st.write('## Prediction')
    st.write('#### Probabilité de défaut de paiment')
    async with aiohttp.ClientSession() as session:
        score = await post(session, dashboard_url+'scoreApplicant', {'applicant': dillEncode(applicant.to_dict('records')[0])})
        if score:
            pass
        else:
            score = _model.predict_proba(applicant.values)[0][1]

        if 'previous_score' not in st.session_state:
            st.session_state.previous_score = score
        score_text = str(round(score*100, 2))+'%'
        if score == st.session_state.previous_score:
            st.metric('', score_text)
        else:
            st.metric('', score_text, delta=str(
                round((score-st.session_state.previous_score)*100, 2))+'%')
        st.session_state.previous_score = score
        st.write('#### Détails')
        shap_explanation = _explainer(applicant)[0]
        st.pyplot(shap.plots.waterfall(
            shap_explanation, max_display=min(nb_features_shap+1, len(list(applicant)))))


async def main(model, explainer, df):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("# Dashboard Prêt à dépenser")
    st.write('## Clients')
    df_filtred = filter_dataframe(df)
    st.dataframe(df_filtred, width=1500)
    filter_viz(df_filtred)

    # Inputs to the sidebar
    st.sidebar.write('# Sélection Client')
    settings_form = st.sidebar.form(key='settings_form')

    # Applicants IDs
    ids = df_filtred['SK_ID_CURR'].values.tolist()
    id = settings_form.selectbox(label='ID client', options=ids)
    idx = ids.index(id)

    # Number of features for SHAP
    nb_features = settings_form.number_input(
        'Nombre de variables (saisie)', 3, len(list(df))-1, 5)
    nb_features_shap = settings_form.number_input(
        'Nombre de variables (détails)', 1, len(list(df))-1, 20)
    settings_form_submit_button = settings_form.form_submit_button('ok')

    # Applicants features
    st.sidebar.write('# Infos Client')
    features_form = st.sidebar.form(key='features_form')

    df_comp = df_filtred.copy()
    df_comp['Clients'] = df_comp.SK_ID_CURR.apply(
        lambda x: 'Sélectionné' if x == id else 'Autres')
    df_comp = df_comp.sort_values(by='Clients')

    applicant = df[df.SK_ID_CURR == id]
    for i in range(nb_features):
        feature = df.drop(
            columns='SK_ID_CURR').columns.tolist()[i]
        min = df[feature].min()
        max = df[feature].max()
        if max == 1:
            value = int(applicant[feature].values[0])
        else:
            value = float(applicant[feature].values[0])
        applicant.loc[applicant.SK_ID_CURR == id, feature] = features_form.number_input(
            feature, min, max, value)
    features_form_submit_button = features_form.form_submit_button('ok')

    if 'load_prediction' not in st.session_state:
        st.session_state.load_prediction = False
    if features_form_submit_button or settings_form_submit_button or st.session_state.load_prediction:
        st.session_state.load_prediction = True
        await applicant_data(applicant)
        compare_viz(df_comp)
        await prediction(applicant, int(
            nb_features_shap), explainer, model)


@cached(ttl=None, cache=Cache.MEMORY)
async def get_data():
    async with aiohttp.ClientSession() as session:
        data = await fetch(session, dashboard_url+'data')
        df = pd.DataFrame.from_dict(data, orient='tight')
        for col in list(df):
            if df[col].max() == 1:
                df[col] = df[col].astype(int)
        return df


@cached(ttl=None, cache=Cache.MEMORY)
async def get_model():
    async with aiohttp.ClientSession() as session:
        model = await fetch(session, dashboard_url+'model')
        if model:
            model = dillDecode(model)
        else:
            st.error(model)
        return model

if __name__ == '__main__':
    # Get data
    df = asyncio.run(get_data())
    # Get the model
    model = asyncio.run(get_model())
    # Compute SHAP values
    explainer = shap.TreeExplainer(model, df.drop(
        columns='SK_ID_CURR'), model_output='probability')

    asyncio.run(main(model, explainer, df))
