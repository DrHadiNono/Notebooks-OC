import streamlit as st
import aiohttp
import asyncio


async def fetch(session, url):
    try:
        async with session.get(url) as response:
            result = await response.json()
            return result
    except Exception:
        return {}


async def post(session, url, data):
    try:
        async with session.post(url, data=data) as response:
            result = await response.json()
            return result
    except Exception:
        return {}


async def main():
    st.write("""
    # Home Credit Dashboard

    Predict if an applicant will encounter payment issues!

    """)

    # async with aiohttp.ClientSession() as session:
    #     data = await fetch(session, 'https://homecredit-api-oc.herokuapp.com/')
    #     if data:
    #         st.write(data)
    #     else:
    #         st.error("Error")

    async with aiohttp.ClientSession() as session:
        # 'https://homecredit-api-oc.herokuapp.com/predict')
        data = await post(session, 'http://127.0.0.1:8000/predict', None)
        if data:
            st.write(data)
        else:
            st.error("Error")

if __name__ == '__main__':
    asyncio.run(main())

# #define the ticker symbol
# tickerSymbol = 'GOOGL'
# #get data on this ticker
# tickerData = yf.Ticker(tickerSymbol)
# #get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# # Open	High	Low	Close	Volume	Dividends	Stock Splits

# st.line_chart(tickerDf.Close)
# st.line_chart(tickerDf.Volume)
