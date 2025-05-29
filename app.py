from flask import Flask, request, jsonify, render_template, make_response
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    url = f'https://query2.finance.yahoo.com/v1/finance/search?q={q}'
    try:
        res = requests.get(url)
        items = res.json().get('quotes', [])
    except Exception:
        return jsonify([])
    results = []
    for it in items:
        sym = it.get('symbol')
        name = it.get('shortname') or it.get('longname') or it.get('exchange')
        if sym:
            results.append({'symbol': sym, 'name': name})
    return jsonify(results)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symbol    = data.get('symbol')
    date_mode = data.get('date_mode')
    interval  = data.get('interval')

    # 計算起迄日期
    if date_mode == 'range':
        start_date = data.get('start_date')
        end_date   = data.get('end_date')
    elif date_mode == 'year':
        years      = int(data.get('years', 1))
        end_date   = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    else:
        start_date = '1990-01-01'
        end_date   = pd.Timestamp.today().strftime('%Y-%m-%d')

    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            return jsonify({'error': '找不到資料，請確認股票代號與日期'}), 400

        # 平整欄位
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [c.strip() for c in df.columns]
        if 'Adj Close' not in df.columns:
            cands = [c for c in df.columns if 'close' in c.lower()]
            df['Adj Close'] = df[cands[0]] if cands else None
            if df['Adj Close'] is None:
                return jsonify({'error': f"無可用的 'Adj Close'，目前欄位：{df.columns.tolist()}"}), 400

        # 指標計算
        df['Return'] = df['Adj Close'].pct_change()
        df['Year']   = df.index.year

        # 每年累積報酬率
        annual_returns = df.groupby('Year')['Return'].apply(lambda r: (r.add(1).prod() - 1))
        avg_return     = float(annual_returns.sum() / len(annual_returns))

        # 每年波動率
        volatility = (df.groupby('Year')['Return'].std() * np.sqrt(252)).round(4)

        # 最大回測
        cum   = (1 + df['Return'].fillna(0)).cumprod()
        dd    = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min().round(4))

        # Sharpe
        sharpe = float(((df['Return'].mean() * 252) / (df['Return'].std() * np.sqrt(252))) if df['Return'].std() else 0)

        # 市場指數
        sym = symbol.upper()
        if sym.endswith('.TW'):
            mkt_ix = '^TWII'
        elif sym.endswith(('.KS', '.KQ')):
            mkt_ix = '^KS11'
        elif sym.endswith('.T'):
            mkt_ix = '^N225'
        elif sym.endswith('.HK'):
            mkt_ix = '^HSI'
        else:
            mkt_ix = '^GSPC'

        mkt = yf.download(mkt_ix, start=start_date, end=end_date, interval=interval)
        mkt.columns = [col[0] if isinstance(col, tuple) else col for col in mkt.columns]
        mkt.columns = [c.strip() for c in mkt.columns]
        retcol = 'Adj Close' if 'Adj Close' in mkt.columns else ('Close' if 'Close' in mkt.columns else None)
        if not retcol:
            return jsonify({'error': f"市場指數 {mkt_ix} 缺失收盤價欄位，目前：{mkt.columns.tolist()}"}), 400
        mkt['Return'] = mkt[retcol].pct_change()

        idx = df['Return'].dropna().index.intersection(mkt['Return'].dropna().index)
        cov = np.cov(df.loc[idx, 'Return'], mkt.loc[idx, 'Return'])
        beta  = float(cov[0,1]/cov[1,1]) if cov[1,1] else None
        alpha = float((df['Return'].mean() - beta*mkt['Return'].mean())*252) if beta else None

        # 繪製 NAV 圖
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(10,4))
        cum.plot(ax=ax)
        ax.set_title(f'{symbol} NAV', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Return (NAV)', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        nav_img = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'volatility'     : volatility.to_dict(),
            'annual_returns' : {int(y): round(v,4) for y,v in annual_returns.items()},
            'avg_return'     : round(avg_return,4),
            'max_drawdown'   : max_dd,
            'sharpe_ratio'   : round(sharpe,4),
            'alpha'          : round(alpha,4) if alpha else None,
            'beta'           : round(beta,4) if beta else None,
            'nav_img'        : nav_img
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export():
    data = request.get_json()
    # 與 analyze 同樣計算日期
    symbol    = data.get('symbol')
    date_mode = data.get('date_mode')
    interval  = data.get('interval')
    if date_mode == 'range':
        start_date = data.get('start_date')
        end_date   = data.get('end_date')
    elif date_mode == 'year':
        years      = int(data.get('years',1))
        end_date   = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    else:
        start_date = '1990-01-01'
        end_date   = pd.Timestamp.today().strftime('%Y-%m-%d')

    # 歷史股價
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df.columns = [col[0] if isinstance(col,tuple) else col for col in df.columns]
    df.columns = [c.strip() for c in df.columns]
    # 確保有 Adj Close
    if 'Adj Close' not in df.columns:
        cands = [c for c in df.columns if 'close' in c.lower()]
        if cands:
            df['Adj Close'] = df[cands[0]]

    # 分析指標
    df['Return'] = df['Adj Close'].pct_change()
    df['Year']   = df.index.year
    annual_returns = df.groupby('Year')['Return'].apply(lambda r: (r.add(1).prod() - 1))
    avg_return     = float(annual_returns.sum() / len(annual_returns))
    volatility     = (df.groupby('Year')['Return'].std() * np.sqrt(252)).round(4)
    cum            = (1 + df['Return'].fillna(0)).cumprod()
    dd             = (cum - cum.cummax()) / cum.cummax()
    max_dd         = float(dd.min().round(4))
    sharpe         = float(((df['Return'].mean() * 252) / (df['Return'].std() * np.sqrt(252))) if df['Return'].std() else 0)

    # 建立 CSV 文字
    hist_csv = df.to_csv(encoding='utf-8-sig')
    lines = []
    lines.append('')
    lines.append('Year,Volatility,AnnualReturn')
    for y in volatility.index:
        lines.append(f'{y},{volatility.loc[y]:.4f},{annual_returns.loc[y]:.4f}')
    lines.append('')
    lines.append(f'AverageReturn,{avg_return:.4f}')
    lines.append(f'MaxDrawdown,{max_dd:.4f}')
    lines.append(f'SharpeRatio,{sharpe:.4f}')
    lines.append(f'Alpha,{alpha:.4f}' if alpha is not None else 'Alpha,')
    lines.append(f'Beta,{beta:.4f}' if beta is not None else 'Beta,')
    metrics_csv = '\n'.join(lines)

    full_csv = hist_csv + '\n' + metrics_csv

    resp = make_response(full_csv)
    resp.headers['Content-Type']        = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = f'attachment; filename={symbol}_history_with_analysis.csv'
    return resp

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
