from gurobipy import *

import pandas as pd
#import pandas.io.data as pull
import pandas_datareader.data as pull
import numpy as np
from math import sqrt

from datetime import datetime
from flask import jsonify, request

stocks = pd.DataFrame.from_csv('./static/data/stocks.csv')
etfs = pd.DataFrame.from_csv('./static/data/etfs.csv')
bonds = pd.DataFrame.from_csv('./static/data/bonds.csv')

def getPortfolio1(df, unused, amount = 1000000,risk = 13, maxP = 5):
    print("Using mean variance for risk calculations")
    T, k = df.shape
    vol = np.cov((df.iloc[1:, :] / df.shift(1).iloc[1:, :]).T) * df.shape[0]
    ret = (np.array(df.tail(1)) / np.array(df.head(1))).ravel()

    data = df.pct_change()[1:]
    total_means = data.mean().values
    syms = data.columns
    sigma = df.cov()
    # Instantiate our model
    m = Model("portfolio")
    portsyms =[]
    etfsyms= []
    bondsyms= []
    for sym in syms:
        if sym[:3] == 'ETF':
            etfsyms.append(sym)
        elif sym[:3] == 'BND':
            bondsyms.append(sym)
        else:
            portsyms.append(sym)
    portvars = [m.addVar(name=symb,lb=0.0) for symb in portsyms]
    portvars = pd.Series(portvars,index=portsyms)
    etfvars = [m.addVar(name=symb,lb=0.0) for symb in etfsyms]
    etfvars = pd.Series(etfvars,index=etfsyms)
    bondvars = [m.addVar(name=symb,lb=0.0) for symb in bondsyms]
    bondvars = pd.Series(bondvars,index=bondsyms)
    # Create one variable for each stock

    allvars = pd.concat([portvars,etfvars,bondvars]).sort_index()
    portfolio = pd.DataFrame({'Variables': allvars})
    m.update()

    # The total budget
    p_total = allvars.sum()

    # The mean return for the portfolio
    p_return = total_means.dot(allvars)

    # The (squared) volatility of the portfolio
    p_risk = sigma.dot(allvars).dot(allvars)


    for var in allvars:
        m.addConstr(var, GRB.LESS_EQUAL, 0.05 *amount)
    m.setObjective(p_return,GRB.MAXIMIZE)

    m.addConstr(p_risk,GRB.LESS_EQUAL, 0.01 * risk * amount * 100000 )
    # m.addConstr(p_risk,GRB.LESS_EQUAL, 0.13 *amount)
    # Fix the budget
    m.addConstr(p_total, GRB.EQUAL, 1000000)


    m.optimize()

    portfolio['stocks'] = allvars.apply(lambda x:x.getAttr('x'))

    pos = portfolio['stocks'].as_matrix()
    pos[np.isclose(pos, 0, atol=1e-3)] = 0
    pos /= np.sum(pos)
    df["pos"] = df.dot(pos)
    perf = df["pos"].to_frame().reset_index()
    perf["index"] = perf["index"].map(lambda d: str(d.date()))
    perf.columns = ["date", "value"]
    allocation = { "L": [{"symbol": s, "p": 0} for s in unused], "S": [],
                   "min": perf["value"].min(),
                   "max": perf["value"].max(),
                   "series": perf.T.to_json() }

    category = lambda x : "S" if x < 0 else "L"

    for i, p in enumerate(pos):
        allocation[category(p)].append({ "symbol": df.columns[i], "p": p })

    allocation["ret"] = (ret.dot(pos) ** ( 365.0 / T ) - 1) * 100

    allocation["vol"] = np.sqrt(pos.dot(vol).dot(pos)) * np.sqrt( 365.0 / T ) * 100
    # allocation["status"] = sol["status"]
    allocation["status"] = "optimal"

    return jsonify(allocation)



def getRebalance1(df, freq, pos):
    return 0


def getFrontier1(df, short):
    T, k = df.shape

    vol = np.cov((df.iloc[1:, :] / df.shift(1).iloc[1:, :]).T) * df.shape[0]
    ret = (np.array(df.tail(1)) / np.array(df.head(1))).ravel()

    sigma = df.cov()
    stats = pd.concat((df.mean(),df.std(),(df+1).prod()-1),axis=1)
    stats.columns = ['Mean_return', 'Volatility', 'Total_return']

    extremes = pd.concat((stats.idxmin(),stats.min(),stats.idxmax(),stats.max()),axis=1)
    extremes.columns = ['Minimizer','Minimum','Maximizer','Maximum']
    growth = (df+1.0).cumprod()
    tx = growth.index
    syms = growth.columns
    # Instantiate our model
    m = Model("portfolio")

    m.setParam('OutputFlag',False)

    # Create one variable for each stock
    portvars = [m.addVar(name=symb,lb=0.0) for symb in syms]
    portvars = pd.Series(portvars, index=syms)
    portfolio = pd.DataFrame({'Variables':portvars})
    m.update()

    # The total budget
    p_total = portvars.sum()

    # The mean return for the portfolio
    p_return = stats['Mean_return'].dot(portvars)

    # The (squared) volatility of the portfolio
    p_risk = sigma.dot(portvars).dot(portvars)

    m.setObjective(p_risk,GRB.MINIMIZE)

    # Fix the budget
    m.addConstr(p_total, GRB.EQUAL, 1)

    m.setParam('Method',1)

    frontier = {}
    fixedreturn = m.addConstr(p_return, GRB.EQUAL, 10)
    m.update()

    # Determine the range of returns. Make sure to include the lowest-risk
    # portfolio in the list of options
    minret = extremes.loc['Mean_return','Minimum']
    maxret = extremes.loc['Mean_return','Maximum']
    riskret = extremes.loc['Volatility','Minimizer']
    riskret = stats.loc[riskret,'Mean_return']
    returns = np.unique(np.hstack((np.linspace(minret,maxret,100),riskret)))

    # Iterate through all returns
    risks = returns.copy()
    for i, alpha in enumerate(returns):
        fixedreturn.rhs = returns[i]
        m.optimize()
        pos = portvars.apply(lambda x:x.getAttr('x')).as_matrix()
        frontier[i] = { "ret": returns[i],
                        "vol": sqrt(p_risk.getValue())}
    return jsonify(frontier)

def getData1():
    raw = pd.read_json(request.form["data"])
    symbols = list(raw.columns)
    data = pd.DataFrame(columns=symbols)
    for symbol in raw.columns:
        idx, val = zip(*list(map(lambda d: (datetime.strptime(d["date"][:10], "%Y-%m-%d").date(), d["value"]), raw[symbol])))
        data[symbol] = pd.Series(data=val, index=pd.DatetimeIndex(idx))
    return data

def pullDataFromYahoo1(symbol, startdate, enddate):
    dates = pd.DatetimeIndex(start=startdate, end=enddate, freq='1d')
    data = pd.DataFrame(index=dates)
    try:
        concatIns = pd.concat([stocks,etfs,bonds],axis=1)
        tmp = concatIns[symbol][startdate:enddate]
        tmp = tmp.to_frame()
        data["value"] = tmp[symbol]
        data = data.interpolate().ffill().bfill()
        data["value"] /= data["value"][0]
        data = data.reset_index()
        data["date"] = data["index"].apply(lambda d: str(d.date()))
        dailyret = np.array(data["value"])
        dailyret = dailyret[1:] / dailyret[:-1]
        return jsonify({ "series": data.drop("index", 1).T.to_json(),
                         "ret": (data["value"].iloc[-1] ** ( 365.0 / data["value"].size ) - 1) * 100,
                         "vol": np.std(dailyret) * 1910.5 })

    except:
        return "invalid"


