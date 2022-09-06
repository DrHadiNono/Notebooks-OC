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


async def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("# Dashboard Prêt à dépenser")

    async with aiohttp.ClientSession() as session:
        data = await fetch(session, dashboard_url+'data')
        df = pd.DataFrame.from_dict(data, orient='tight')
        st.dataframe(df, width=1500)

    async with aiohttp.ClientSession() as session:
        expected_value = await fetch(session, dashboard_url+'expectedValue')
        if expected_value:
            expected_value = dillDecode(expected_value)
        else:
            st.error(expected_value)

    st.write('## Client')

    ids = []
    async with aiohttp.ClientSession() as session:
        ids = await fetch(session, dashboard_url+'ids')
    id = st.selectbox(label='ID client', options=ids)
    idx = ids.index(id)

    applicant = df[df.SK_ID_CURR == id]
    feature = st.selectbox(label='Valeur', options=df.drop(
        columns='SK_ID_CURR').columns)
    min = df[feature].min()
    max = df[feature].max()
    if max == 1:
        step = 1
        value = int(applicant[feature].values[0])
    else:
        step = 0.01
        value = float(applicant[feature].values[0])

    applicant.loc[applicant.SK_ID_CURR == id, feature] = st.number_input(
        '', min, max, value)

    st.dataframe(applicant, width=1500)

    st.write('## Prediction')
    st.write('#### Probabilité de défaut de paiment')
    async with aiohttp.ClientSession() as session:
        score = await post(session, dashboard_url+'scoreApplicant', {'applicant': dillEncode(applicant.to_dict('records')[0])})
        if score:
            st.metric('',
                      str(round(score*100, 2))+'%')
        else:
            st.error(score)

    st.write('#### Détails')
    nb_features = st.number_input(
        'Nombre de variables', 1, len(list(df))-1, 20)
    async with aiohttp.ClientSession() as session:
        shap_explanation = await post(session, dashboard_url+'shapExplanationApplicant', {'applicant': dillEncode(applicant.to_dict())})
        if shap_explanation:
            shap_explanation = dillDecode(shap_explanation)
            st.pyplot(shap.plots.waterfall(
                shap_explanation, max_display=nb_features))
        else:
            st.error(shap_explanation)

    # async with aiohttp.ClientSession() as session:
    #     shap_values = await fetch(session, dashboard_url+'shapValues')
    #     if shap_values:
    #         shap_values = dillDecode(shap_values)
    #         st.pyplot(shap.summary_plot(shap_values, df.drop(
    #             columns='SK_ID_CURR'), plot_type="bar", max_display=20))
    #     else:
    #         st.error(shap_values)

    # async with aiohttp.ClientSession() as session:
    #     shap_value = await post(session, dashboard_url+'shapValue', {'idx': idx})
    #     if shap_value:
    #         shap_value = dillDecode(shap_value)
    #         st.pyplot(shap.plots.force(expected_value, shap_value,
    #                   df.iloc[idx, 1:], feature_names=df.columns.tolist()[1:], matplotlib=True))
    #     else:
    #         st.error(shap_value)


if __name__ == '__main__':
    asyncio.run(main())
