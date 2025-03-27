import sys
import os
import warnings
from flask import Flask, request, jsonify

# Suprimir FutureWarning do huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
# Suprimir UserWarning do torch (opcional, se desejar)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TA_Lib import TradingStrategy, print_results
from market_data import get_stock_data
from news_scraper import get_financial_news
from sentiment_analysis import analyze_sentiment
from gemma_chat import query_gemma

app = Flask(__name__)

# Armazenar os resultados mais recentes para reutilização na rota /detailed
latest_analysis = {}

def analyze_with_tars(ticker, df_1d, df_1h, df_5m, final_capital_1d, signal_1d, prob_1d, action_1h, prob_1h, action_5m, prob_5m, sentiment, pct_change_1d, pct_change_1h, pct_change_5m, news_analyzed, sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d, sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h, sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m):
    if df_1d is None or df_1h is None or df_5m is None or df_1d.empty or df_1h.empty or df_5m.empty:
        return "Erro: Um ou mais DataFrames (1d, 1h, 5m) estão indisponíveis ou vazios para a análise do TARS."
    
    final_capital_1d = final_capital_1d if final_capital_1d is not None else 10000.0
    signal_1d = signal_1d if signal_1d is not None else "Neutro"
    prob_1d = prob_1d if prob_1d is not None else 0.0
    action_1h = action_1h if action_1h is not None else "Neutro"
    prob_1h = prob_1h if prob_1h is not None else 0.0
    action_5m = action_5m if action_5m is not None else "Neutro"
    prob_5m = prob_5m if prob_5m is not None else 0.0
    pct_change_1d = pct_change_1d if pct_change_1d is not None else 0.0
    pct_change_1h = pct_change_1h if pct_change_1h is not None else 0.0
    pct_change_5m = pct_change_5m if pct_change_5m is not None else 0.0
    sentiment = sentiment if sentiment is not None else "Neutro"
    news_analyzed = news_analyzed if news_analyzed else [("Nenhuma notícia encontrada", {"label": "neutral"})]

    volumes_1d = df_1d['Volume'].tail(30).tolist()
    volumes_1h = df_1h['Volume'].tail(30).tolist()
    volumes_5m = df_5m['Volume'].tail(30).tolist()

    # Obter o preço atual
    current_price = float(df_1d['Close'].iloc[-1]) if df_1d is not None and not df_1d.empty else 0.0

    news_text = "\n".join([f"{i+1}. {headline} - Sentimento: {sentiment['label'].capitalize()}" for i, (headline, sentiment) in enumerate(news_analyzed)])

    try:
        prompt = (
            f"Analise o ativo {ticker} com base nos seguintes dados:\n"
            f"- Preço atual: ${current_price:.2f}\n"
            f"- Período 1d (10 anos): Capital final: ${final_capital_1d:.2f}, Sinal: {signal_1d}, Probabilidade: {prob_1d:.2f}%, Previsão de Variação: {pct_change_1d*100:.2f}%\n"
            f"- Período 1h (1 ano): Sinal: {action_1h}, Probabilidade: {prob_1h:.2f}%, Previsão de Variação: {pct_change_1h*100:.2f}%\n"
            f"- Período 5m (1 mês): Sinal: {action_5m}, Probabilidade: {prob_5m:.2f}%, Previsão de Variação: {pct_change_5m*100:.2f}%\n"
            f"- Volumes recentes (1d, últimos 30 períodos): {', '.join([f'{v:.2f}' for v in volumes_1d])}\n"
            f"- Volumes recentes (1h, últimos 30 períodos): {', '.join([f'{v:.2f}' for v in volumes_1h])}\n"
            f"- Volumes recentes (5m, últimos 30 períodos): {', '.join([f'{v:.2f}' for v in volumes_5m])}\n"
            f"- Notícias do mercado:\n{news_text}\nClassificação Geral do Sentimento: {sentiment}\n"
            f"Com base nisso, qual é a sua recomendação para o ativo {ticker}? Forneça uma análise detalhada. "
            f"Se houver sinal de 'Comprar' ou 'Vender' em qualquer período, inclua a entrada, stop gain, stop loss, probabilidade de acerto e previsão de variação correspondentes. "
            f"Certifique-se de que os valores de entrada, stop gain e stop loss sejam consistentes com o preço atual (${current_price:.2f})."
        )
        response = query_gemma(prompt)
        return f"Análise do TARSg:\n{response}"
    except Exception as e:
        return f"Erro: Não foi possível consultar o modelo TARSg. Detalhes: {str(e)}"

def analyze_period(ticker, df, interval, signal, prob, pct_change, sl_buy, sg_buy, sl_sell, sg_sell, volumes, news_analyzed, sentiment):
    try:
        news_text = "\n".join([f"{i+1}. {headline} - Sentimento: {sentiment['label'].capitalize()}" for i, (headline, sentiment) in enumerate(news_analyzed)])
        prompt = (
            f"Analise o ativo {ticker} apenas para o período {interval}:\n"
            f"- Sinal: {signal}, Probabilidade: {prob:.2f}%, Previsão de Variação: {pct_change*100:.2f}%\n"
            f"- Volumes recentes (últimos 30 períodos): {', '.join([f'{v:.2f}' for v in volumes])}\n"
            f"- Notícias do mercado:\n{news_text}\nClassificação Geral do Sentimento: {sentiment}\n"
            f"Forneça uma análise detalhada para este período. Se o sinal for 'Comprar' ou 'Vender', inclua a entrada, stop gain, stop loss, probabilidade de acerto e previsão de variação."
        )
        response = query_gemma(prompt)
        return f"Análise do TARSg para {interval}:\n{response}"
    except Exception as e:
        return f"Erro: Não foi possível consultar o modelo TARSg para {interval}. Detalhes: {str(e)}"

def calculate_stops(current_price, signal, pct_change):
    """
    Calcula Stop Loss e Stop Gain com base no preço atual, sinal e previsão de variação.
    Retorna valores arredondados para 3 casas decimais.
    """
    if current_price is None or current_price <= 0:
        return "N/A", "N/A", "N/A", "N/A"

    # Converter pct_change para decimal (ex.: 1.232% -> 0.01232)
    pct_change_decimal = pct_change / 100

    if signal == "Comprar":
        # Para compra: Stop Gain é maior que o preço atual, Stop Loss é menor
        expected_price = current_price * (1 + pct_change_decimal)
        stop_gain_buy = expected_price * 1.02  # 2% acima do preço esperado
        stop_loss_buy = current_price * 0.98   # 2% abaixo do preço atual
        stop_loss_sell = "N/A"
        stop_gain_sell = "N/A"
    elif signal == "Vender":
        # Para venda: Stop Gain é menor que o preço atual, Stop Loss é maior
        expected_price = current_price * (1 - pct_change_decimal)
        stop_gain_sell = expected_price * 0.98  # 2% abaixo do preço esperado
        stop_loss_sell = current_price * 1.02   # 2% acima do preço atual
        stop_loss_buy = "N/A"
        stop_gain_buy = "N/A"
    else:
        # Para "Neutro" ou "Mantenha"
        stop_loss_buy = "N/A"
        stop_gain_buy = "N/A"
        stop_loss_sell = "N/A"
        stop_gain_sell = "N/A"

    # Arredondar para 3 casas decimais
    stop_loss_buy = round(float(stop_loss_buy), 3) if stop_loss_buy != "N/A" else "N/A"
    stop_gain_buy = round(float(stop_gain_buy), 3) if stop_gain_buy != "N/A" else "N/A"
    stop_loss_sell = round(float(stop_loss_sell), 3) if stop_loss_sell != "N/A" else "N/A"
    stop_gain_sell = round(float(stop_gain_sell), 3) if stop_gain_sell != "N/A" else "N/A"

    return stop_loss_buy, stop_gain_buy, stop_loss_sell, stop_gain_sell

@app.route('/analyze', methods=['POST'])
def analyze_ticker():
    global latest_analysis
    ticker = request.form['ticker'].strip().upper()
    print(f"API: Processando ticker {ticker}")
    
    if not ticker:
        return jsonify({"error": "Nenhum ticker fornecido."}), 400
    
    if '.SA' in ticker or ticker.endswith(('3', '4', '5', '6', '11')):
        ticker = ticker.replace('.SA', '') + '.SA'
        is_brazilian = True
    else:
        is_brazilian = False
    
    ts = TradingStrategy()
    periods = [
        ('10y', '1d'),
        ('1y', '1h'),
        ('1mo', '5m')
    ]
    
    df_1d, df_1h, df_5m = None, None, None
    final_capital_1d, signal_1d, prob_1d, action_1h, prob_1h, action_5m, prob_5m = None, None, None, None, None, None, None
    pct_change_1d, pct_change_1h, pct_change_5m = None, None, None
    sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d = None, None, None, None
    sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h = None, None, None, None
    sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m = None, None, None, None
    
    try:
        for period, interval in periods:
            df = ts.get_data(ticker, period=period, interval=interval)
            if df.empty:
                print(f"API: Dados não disponíveis para {ticker} no período {interval}.")
                continue
            
            if interval == '1d':
                final_capital_1d, trades_1d, df_1d, _, _ = ts.backtest(df)
                if final_capital_1d is None:
                    print("API: Falha no backtest para o período 1d.")
                    continue
                signal_1d, prob_1d, sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d, pct_change_1d = ts.predict_action(df, ticker, '1d')
            else:
                features, targets, scaler = ts.prepare_data_for_xgb(df, interval)
                if features.empty or not targets:
                    print(f"API: Dados insuficientes para treinar o modelo XGBoost no período {interval}.")
                    continue
                ts.train_xgb_model(features, targets, interval)
                if interval == '1h':
                    df_1h = df
                    action_1h, prob_1h, sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h, pct_change_1h = ts.predict_action(df, ticker, '1h')
                elif interval == '5m':
                    df_5m = df
                    action_5m, prob_5m, sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m, pct_change_5m = ts.predict_action(df, ticker, '5m')
        
        news = get_financial_news(ticker, is_brazilian)
        if not news or "Erro" in news[0]:
            sentiment = "Neutro"
            news_analyzed = [("Nenhuma notícia encontrada", {"label": "neutral"})]
            news_summary = "Notícias e Sentimentos: Nenhuma notícia encontrada.\nClassificação Geral: Neutro"
        else:
            news_analyzed = [(n, analyze_sentiment(n)) for n in news[:10]]
            news_lines = [f"{i}. {headline} - {sentiment['label'].capitalize()}" for i, (headline, sentiment) in enumerate(news_analyzed, 1)]
            total_positive = sum(2 if ticker.replace(".SA", "") in headline else 1 for headline, sentiment in news_analyzed if sentiment['label'] == 'positive')
            total_negative = sum(2 if ticker.replace(".SA", "") in headline else 1 for headline, sentiment in news_analyzed if sentiment['label'] == 'negative')
            total_neutral = sum(2 if ticker.replace(".SA", "") in headline else 1 for headline, sentiment in news_analyzed if sentiment['label'] == 'neutral')
            sentiment = "Positivo" if total_positive > total_negative and total_positive > total_neutral else "Negativo" if total_negative > total_positive and total_negative > total_neutral else "Neutro"
            news_summary = "Notícias e Sentimentos:\n" + "\n".join(news_lines) + f"\nClassificação Geral: {sentiment}"
        
        tars_analysis = analyze_with_tars(ticker, df_1d, df_1h, df_5m, final_capital_1d, signal_1d, prob_1d, 
                                         action_1h, prob_1h, action_5m, prob_5m, sentiment,
                                         pct_change_1d, pct_change_1h, pct_change_5m, news_analyzed,
                                         sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d,
                                         sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h,
                                         sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m)
        
        # Garantir que os valores sejam preenchidos com padrões se forem None e arredondar para 3 casas decimais
        final_capital_1d = round(float(final_capital_1d), 3) if final_capital_1d is not None else 10000.0
        signal_1d = signal_1d if signal_1d is not None else "Neutro"
        prob_1d = round(float(prob_1d), 3) if prob_1d is not None else 0.0
        action_1h = action_1h if action_1h is not None else "Neutro"
        prob_1h = round(float(prob_1h), 3) if prob_1h is not None else 0.0
        action_5m = action_5m if action_5m is not None else "Neutro"
        prob_5m = round(float(prob_5m), 3) if prob_5m is not None else 0.0
        pct_change_1d = round(float(pct_change_1d), 3) if pct_change_1d is not None else 0.0
        pct_change_1h = round(float(pct_change_1h), 3) if pct_change_1h is not None else 0.0
        pct_change_5m = round(float(pct_change_5m), 3) if pct_change_5m is not None else 0.0

        # Obter o preço atual para cada período
        current_price_1d = float(df_1d['Close'].iloc[-1]) if df_1d is not None and not df_1d.empty else None
        current_price_1h = float(df_1h['Close'].iloc[-1]) if df_1h is not None and not df_1h.empty else None
        current_price_5m = float(df_5m['Close'].iloc[-1]) if df_5m is not None and not df_5m.empty else None

        # Recalcular Stop Loss e Stop Gain com base no preço atual e na previsão de variação
        sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d = calculate_stops(current_price_1d, signal_1d, pct_change_1d * 100)
        sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h = calculate_stops(current_price_1h, action_1h, pct_change_1h * 100)
        sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m = calculate_stops(current_price_5m, action_5m, pct_change_5m * 100)

        # Armazenar os dados para uso na rota /detailed
        latest_analysis[ticker] = {
            '1d': {
                'initial_capital': 10000.0,
                'final_capital': final_capital_1d,
                'signal': signal_1d,
                'prob': prob_1d,
                'pct_change': round(pct_change_1d * 100, 3),
                'sl_buy': sl_buy_1d,
                'sg_buy': sg_buy_1d,
                'sl_sell': sl_sell_1d,
                'sg_sell': sg_sell_1d,
            },
            '1h': {
                'action': action_1h,
                'prob': prob_1h,
                'pct_change': round(pct_change_1h * 100, 3),
                'sl_buy': sl_buy_1h,
                'sg_buy': sg_buy_1h,
                'sl_sell': sl_sell_1h,
                'sg_sell': sg_sell_1h,
            },
            '5m': {
                'action': action_5m,
                'prob': prob_5m,
                'pct_change': round(pct_change_5m * 100, 3),
                'sl_buy': sl_buy_5m,
                'sg_buy': sg_buy_5m,
                'sl_sell': sl_sell_5m,
                'sg_sell': sg_sell_5m,
            }
        }
        
        # Log para depuração
        print(f"Dados armazenados em latest_analysis para {ticker}:")
        print(f"1d: {latest_analysis[ticker]['1d']}")
        print(f"1h: {latest_analysis[ticker]['1h']}")
        print(f"5m: {latest_analysis[ticker]['5m']}")
        
        results = {
            'ticker': ticker,
            'close': round(float(df_1d['Close'].iloc[-1]), 3) if df_1d is not None and not df_1d.empty else 0.0,
            'signal_1d': signal_1d,
            'prob_1d': prob_1d,
            'pct_change_1d': round(pct_change_1d * 100, 3),
            'action_1h': action_1h,
            'prob_1h': prob_1h,
            'pct_change_1h': round(pct_change_1h * 100, 3),
            'action_5m': action_5m,
            'prob_5m': prob_5m,
            'pct_change_5m': round(pct_change_5m * 100, 3),
            'backtest_return': round(float((final_capital_1d - 10000) / 10000 * 100), 3) if final_capital_1d else 0.0,
            'volume_1d': ('acima' if df_1d['Volume'].iloc[-1] > df_1d['Volume_EMA_7'].iloc[-1] else 'abaixo') + ' da EMA 7, ' + \
                         ('acima' if df_1d['Volume'].iloc[-1] > df_1d['Volume_EMA_30'].iloc[-1] else 'abaixo') + ' da EMA 30' if df_1d is not None and not df_1d.empty else "N/A",
            'volume_1h': ('acima' if df_1h['Volume'].iloc[-1] > df_1h['Volume_EMA_7'].iloc[-1] else 'abaixo') + ' da EMA 7, ' + \
                         ('acima' if df_1h['Volume'].iloc[-1] > df_1h['Volume_EMA_30'].iloc[-1] else 'abaixo') + ' da EMA 30' if df_1h is not None and not df_1h.empty else "N/A",
            'volume_5m': ('acima' if df_5m['Volume'].iloc[-1] > df_5m['Volume_EMA_7'].iloc[-1] else 'abaixo') + ' da EMA 7, ' + \
                         ('acima' if df_5m['Volume'].iloc[-1] > df_5m['Volume_EMA_30'].iloc[-1] else 'abaixo') + ' da EMA 30' if df_5m is not None and not df_5m.empty else "N/A",
            'news_summary': news_summary,
            'tars_analysis': tars_analysis
        }
        print(f"API: Retornando resultados para {ticker}")
        return jsonify(results)
    
    except Exception as e:
        print(f"API: Erro ao processar {ticker}: {str(e)}")
        return jsonify({"error": f"Erro ao processar {ticker}: {str(e)}"}), 500

@app.route('/detailed', methods=['POST'])
def get_detailed():
    ticker = request.form['ticker'].strip().upper()
    period = request.form['period'].strip().lower()
    print(f"API: Solicitando detalhes para {ticker}, período {period}")
    
    if ticker not in latest_analysis:
        return jsonify({"error": f"Análise para {ticker} não encontrada. Por favor, analise o ticker primeiro."}), 404
    
    if period not in latest_analysis[ticker]:
        return jsonify({"error": f"Período {period} inválido. Use '1d', '1h' ou '5m'."}), 400
    
    details = latest_analysis[ticker][period]
    return jsonify(details)

def main():
    print("Você pode consultar qualquer ativo da bolsa americana (ex.: AAPL, TSLA) ou brasileira (ex.: PETR4.SA, WEGE3.SA).")
    print("Para ações brasileiras, inclua ou não o '.SA' no final; o sistema ajustará automaticamente.")
    
    while True:
        ticker = input("Qual ativo deseja consultar? ").strip().upper()
        if not ticker:
            print("Nenhum ticker fornecido. Tente novamente.")
            continue
        
        if '.SA' in ticker or ticker.endswith(('3', '4', '5', '6', '11')):
            ticker = ticker.replace('.SA', '') + '.SA'
            print(f"Ativo brasileiro identificado: {ticker}")
            is_brazilian = True
        else:
            print(f"Ativo americano identificado: {ticker}")
            is_brazilian = False
        
        ts = TradingStrategy()
        periods = [
            ('10y', '1d'),
            ('1y', '1h'),
            ('1mo', '5m')
        ]
        
        df_1d, df_1h, df_5m = None, None, None
        final_capital_1d, signal_1d, prob_1d, action_1h, prob_1h, action_5m, prob_5m = None, None, None, None, None, None, None
        pct_change_1d, pct_change_1h, pct_change_5m = None, None, None
        sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d = None, None, None, None
        sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h = None, None, None, None
        sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m = None, None, None, None
        
        try:
            for period, interval in periods:
                print(f"\nPeríodo: {period}, Intervalo: {interval}")
                df = ts.get_data(ticker, period=period, interval=interval)
                if df.empty:
                    print(f"Erro: Dados não disponíveis para {ticker} no período {interval}.")
                    continue
                
                if interval == '1d':
                    final_capital_1d, trades_1d, df_1d, _, _ = ts.backtest(df)
                    if final_capital_1d is None:
                        print("Erro: Falha no backtest para o período 1d.")
                        continue
                    signal_1d, prob_1d, sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d, pct_change_1d = ts.predict_action(df, ticker, '1d')
                    print_results(10000, final_capital_1d, trades_1d, signal_1d, prob_1d, sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d, pct_change_1d)
                else:
                    try:
                        features, targets, scaler = ts.prepare_data_for_xgb(df, interval)
                        if features.empty or not targets:
                            print(f"Erro: Dados insuficientes para treinar o modelo XGBoost no período {interval}.")
                            continue
                        ts.train_xgb_model(features, targets, interval)
                        if interval == '1h':
                            df_1h = df
                            action_1h, prob_1h, sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h, pct_change_1h = ts.predict_action(df, ticker, '1h')
                            print_results(10000, 10000, [], action_1h, prob_1h, sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h, pct_change_1h)
                        elif interval == '5m':
                            df_5m = df
                            action_5m, prob_5m, sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m, pct_change_5m = ts.predict_action(df, ticker, '5m')
                            print_results(10000, 10000, [], action_5m, prob_5m, sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m, pct_change_5m)
                    except ValueError as e:
                        print(f"Erro ao processar dados para o intervalo {interval}: {str(e)}")
                        continue
            
            print("\nBuscando notícias...")
            try:
                news = get_financial_news(ticker, is_brazilian)
                if not news or "Erro" in news[0]:
                    sentiment = "Neutro"
                    news_analyzed = [("Nenhuma notícia encontrada", {"label": "neutral"})]
                    news_summary = "Notícias e Sentimentos: Nenhuma notícia encontrada.\nClassificação Geral: Neutro"
                else:
                    news_analyzed = [(n, analyze_sentiment(n)) for n in news[:10]]
                    news_lines = [f"{i}. {headline} - {sentiment['label'].capitalize()}" for i, (headline, sentiment) in enumerate(news_analyzed, 1)]
                    total_positive = sum(2 if ticker.replace(".SA", "") in headline else 1 for headline, sentiment in news_analyzed if sentiment['label'] == 'positive')
                    total_negative = sum(2 if ticker.replace(".SA", "") in headline else 1 for headline, sentiment in news_analyzed if sentiment['label'] == 'negative')
                    total_neutral = sum(2 if ticker.replace(".SA", "") in headline else 1 for headline, sentiment in news_analyzed if sentiment['label'] == 'neutral')
                    sentiment = "Positivo" if total_positive > total_negative and total_positive > total_neutral else "Negativo" if total_negative > total_positive and total_negative > total_neutral else "Neutro"
                    news_summary = "Notícias e Sentimentos:\n" + "\n".join(news_lines) + f"\nClassificação Geral: {sentiment}"
                    print(news_summary)
            except Exception as e:
                print(f"Erro ao buscar notícias ou analisar sentimento: {e}")
                sentiment = "Neutro"
                news_analyzed = [("Erro ao buscar notícias", {"label": "neutral"})]
                news_summary = "Notícias e Sentimentos: Erro ao buscar notícias.\nClassificação Geral: Neutro"
            
            print("\nConsultando TARSg...")
            tars_analysis = analyze_with_tars(ticker, df_1d, df_1h, df_5m, final_capital_1d, signal_1d, prob_1d, 
                                             action_1h, prob_1h, action_5m, prob_5m, sentiment,
                                             pct_change_1d, pct_change_1h, pct_change_5m, news_analyzed,
                                             sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d,
                                             sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h,
                                             sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m)
            print(tars_analysis)
        
        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
        
        while True:
            print("\nO que gostaria de fazer agora?")
            print("1. Consultar outra empresa")
            print("2. Ver outro período deste ativo")
            print("3. Ver previsões separadas")
            print("4. Perguntar algo mais sobre a ação")
            print("5. Sair")
            choice = input("Digite o número da sua escolha (1-5): ").strip()
            
            if choice == '1':
                break
            elif choice == '2':
                print("Função de outro período ainda não implementada. Escolha outra opção.")
                continue
            elif choice == '3':
                period_choice = input("Qual período deseja analisar? (1d, 1h, 5m): ").strip().lower()
                if period_choice == '1d' and df_1d is not None and not df_1d.empty:
                    volumes_1d = df_1d['Volume'].tail(30).tolist()
                    print(analyze_period(ticker, df_1d, '1d', signal_1d, prob_1d, pct_change_1d, sl_buy_1d, sg_buy_1d, sl_sell_1d, sg_sell_1d, volumes_1d, news_analyzed, sentiment))
                elif period_choice == '1h' and df_1h is not None and not df_1h.empty:
                    volumes_1h = df_1h['Volume'].tail(30).tolist()
                    print(analyze_period(ticker, df_1h, '1h', action_1h, prob_1h, pct_change_1h, sl_buy_1h, sg_buy_1h, sl_sell_1h, sg_sell_1h, volumes_1h, news_analyzed, sentiment))
                elif period_choice == '5m' and df_5m is not None and not df_5m.empty:
                    volumes_5m = df_5m['Volume'].tail(30).tolist()
                    print(analyze_period(ticker, df_5m, '5m', action_5m, prob_5m, pct_change_5m, sl_buy_5m, sg_buy_5m, sl_sell_5m, sg_sell_5m, volumes_5m, news_analyzed, sentiment))
                else:
                    print("Período inválido ou dados indisponíveis. Escolha '1d', '1h' ou '5m'.")
                continue
            elif choice == '4':
                question = input(f"Digite sua pergunta sobre {ticker}: ").strip()
                if question:
                    prompt = f"Pergunta sobre o ativo {ticker}: {question}\nForneça uma resposta detalhada com base nos dados disponíveis."
                    response = query_gemma(prompt)
                    print(f"Resposta do TARSg:\n{response}")
                else:
                    print("Nenhuma pergunta fornecida.")
                continue
            elif choice == '5':
                print("Saindo...")
                sys.exit(0)
            else:
                print("Opção inválida. Digite um número entre 1 e 5.")
                continue

if __name__ == "__main__":
   app.run(host='0.0.0.0', debug=True, port=5001)