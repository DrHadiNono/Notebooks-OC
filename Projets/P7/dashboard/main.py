import streamlit as st
import aiohttp
import asyncio
import pandas as pd
import dill
import codecs
import shap
# import traceback

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

# @st.experimental_memo(suppress_st_warning=True)


async def load_applicant(applicant, nb_features_shap, explainer, model):
    st.write('## Client Sélectionné')
    st.dataframe(applicant, width=1500)

    st.write('## Prediction')
    st.write('#### Probabilité de défaut de paiment')
    async with aiohttp.ClientSession() as session:
        score = await post(session, dashboard_url+'scoreApplicant', {'applicant': dillEncode(applicant.to_dict('records')[0])})
        if score:
            pass
        else:
            score = model.predict_proba(applicant.values)[0][1]
        st.metric('',
                  str(round(score*100, 2))+'%')

        st.write('#### Détails')
        shap_explanation = explainer(applicant)[0]
        st.pyplot(shap.plots.waterfall(
            shap_explanation, max_display=min(nb_features_shap+1, len(list(applicant)))))

        # shap_explanation = await post(session, dashboard_url+'shapExplanationApplicant', {'applicant': dillEncode(applicant.to_dict())})
        # if shap_explanation:
        #     shap_explanation = dillDecode(shap_explanation)
        #     st.pyplot(shap.plots.waterfall(
        #         shap_explanation, max_display=nb_features_shap))
        # else:
        #     st.error(shap_explanation)


async def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    async with aiohttp.ClientSession() as session:
        # Get data
        data = await fetch(session, dashboard_url+'data')
        df = pd.DataFrame.from_dict(data, orient='tight')
        for col in list(df):
            if df[col].max() == 1:
                df[col] = df[col].astype(int)
        st.write("# Dashboard Prêt à dépenser")
        st.write('## Clients')
        st.dataframe(df, width=1500)

        # Get the model
        model = await fetch(session, dashboard_url+'model')
        if model:
            model = dillDecode(model)
        else:
            st.error(model)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model, df.drop(
            columns='SK_ID_CURR'), model_output='probability')
        expected_value = explainer.expected_value

        # Gat SHAP expected_value
        # expected_value = await fetch(session, dashboard_url+'expectedValue')
        # if expected_value:
        #     expected_value = dillDecode(expected_value)
        # else:
        #     st.error(expected_value)

    # Inputs to the sidebar

    settings_form = st.sidebar.form(key='settings_form')
    # Applicants IDs
    ids = []
    async with aiohttp.ClientSession() as session:
        ids = await fetch(session, dashboard_url+'ids')
    id = settings_form.selectbox(label='ID client', options=ids)
    idx = ids.index(id)

    # Number of features for SHAP
    nb_features = settings_form.number_input(
        'Nombre de variables (saisie)', 1, len(list(df))-1, 5)
    nb_features_shap = settings_form.number_input(
        'Nombre de variables (détails)', 1, len(list(df))-1, 20)
    settings_form_submit_button = settings_form.form_submit_button('ok')

    features_form = st.sidebar.form(key='features_form')
    # Applicants features
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

    await load_applicant(applicant.drop(columns='SK_ID_CURR'), nb_features_shap, explainer, model)


if __name__ == '__main__':
    asyncio.run(main())
