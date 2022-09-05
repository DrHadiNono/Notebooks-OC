from socket import socket
import streamlit as st
import aiohttp
import asyncio
# import traceback

dashboard_url = 'https://homecredit-api-oc.herokuapp.com/'
# dashboard_url = 'http://127.0.0.1:8000/'


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
    st.write("""
    # Home Credit Dashboard

    Predict if an applicant will encounter payment issues!

    """)

    ids = []
    async with aiohttp.ClientSession() as session:
        ids = await fetch(session, dashboard_url+'ids')
    id = st.selectbox(label='ID client', options=ids)

    async with aiohttp.ClientSession() as session:
        score = await post(session, dashboard_url+'predict', {'id': id})
        if score:
            st.metric(label="Score", value=str(round(score*100, 2))+'%')
        else:
            st.error(score)

if __name__ == '__main__':
    asyncio.run(main())
