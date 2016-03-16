
import numpy as np
import pandas as pd

from datetime import datetime
from flask import Flask, render_template, request

from static.py.meanvariance import getData1, getPortfolio1, getFrontier1, getRebalance1, pullDataFromYahoo1
from static.py.mad import getData, getPortfolio, getFrontier, getRebalance,pullDataFromYahoo

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/', methods=['GET', 'POST'])
def _index():
    return render_template('index.html')

@app.route('/_pull', methods=['POST'])
def _pullData():
    symbol = request.form["symbol"].strip()
    startdate = datetime.strptime(request.form["startdate"], "%Y-%m-%d").date()
    enddate = datetime.strptime(request.form["enddate"], "%Y-%m-%d").date()

    return pullDataFromYahoo(symbol, startdate, enddate)
    
@app.route('/_fit', methods=['POST'])
def _fitModel():
    risk = float(request.form["risk"])
    totalAmount = float(request.form["totinvest"])
    maxinvest = float(request.form["maxinvest"])
    model = int(request.form["model"])

    unused = filter(lambda s: len(s) > 0, request.form["unused"].split(","))
    data = getData()
    if model == 102:
        return getPortfolio1(data, unused, amount=totalAmount,risk=risk, maxP=maxinvest)
    else:
        return getPortfolio(data, unused, amount=totalAmount,risk=risk, maxP=maxinvest)

@app.route('/_frontier', methods=['POST'])
def _fitFrontier():
    short = request.form["shor"] == "true"
    df = getData()
        
    return getFrontier(df, short)
    
@app.route('/_rebalancing', methods=['POST'])
def _rebalancing():
    df = getData()
    pos = { row["symbol"]: row["p"] for _, row in pd.read_json(request.form["pos"]).iterrows() }
    pos = np.array([pos[s] for s in df.columns])
    freq = str(request.form["rbfreq"]) + "m"
    
    return getRebalance(df, freq, pos)
    
if __name__ == '__main__':
    app.run(debug=True,port=12345)
