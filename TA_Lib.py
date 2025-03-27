import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class TradingStrategy:
    def __init__(self):
        self.media_curta = 9
        self.media_longa = 51
        self.media_super_longa = 200
        self.open_trades = {}
        self.xgb_model_1d = None
        self.xgb_model = None
        self.scalers = {}  # Dicionário para armazenar os scalers por intervalo

    def get_data(self, ticker, period='10y', interval='1d'):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                print(f"API: Dados não disponíveis para {ticker} no período {interval}.")
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            print(f"API: Data final para {ticker} ({interval}): {df.index[-1]}")
            return df
        except Exception as e:
            print(f"Erro ao baixar dados para {ticker}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df, interval='1d'):
        if df.empty:
            return df
        
        df['EMA_Curta'] = df['Close'].ewm(span=self.media_curta, adjust=False).mean()
        df['EMA_Longa'] = df['Close'].ewm(span=self.media_longa, adjust=False).mean()
        df['EMA_Curta_Slope'] = df['EMA_Curta'].diff()
        df['EMA_Longa_Slope'] = df['EMA_Longa'].diff()
        df['Volume_EMA_7'] = df['Volume'].ewm(span=7, adjust=False).mean()
        df['Volume_EMA_30'] = df['Volume'].ewm(span=30, adjust=False).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Ajuste dinâmico da janela do ATR
        atr_window = 14  # Padrão, ajustado por intervalo
        if interval == '1d':
            atr_window = 14  # 14 dias
        elif interval == '1h':
            atr_window = 14  # 14 horas
        elif interval == '5m':
            atr_window = 14  # 14 candles de 5 minutos
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=atr_window).mean()

        if interval == '1d' and self.media_super_longa > 0:
            df['EMA_Super_Longa'] = df['Close'].ewm(span=self.media_super_longa, adjust=False).mean()
            df['EMA_SL_Slope'] = df['EMA_Super_Longa'].diff()
        
        if interval in ['1h', '5m']:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

        return df

    def backtest(self, df, use_super_longa=True):
        if df.empty:
            return 10000, [], df, pd.DataFrame(), []
        df = self.calculate_indicators(df, interval='1d')
        capital = 10000
        position = 0
        entry_price = 0
        trades = []
        features_list = []
        targets = []
        look_ahead = 5

        for i in range(1, len(df) - look_ahead):
            close = df['Close'].iloc[i]
            future_close = df['Close'].iloc[i + look_ahead]
            pct_change = (future_close - close) / close
            rsi = df['RSI'].iloc[i]
            if pd.isna(rsi):
                continue

            features = {
                'EMA_Curta': df['EMA_Curta'].iloc[i],
                'EMA_Longa': df['EMA_Longa'].iloc[i],
                'EMA_Super_Longa': df['EMA_Super_Longa'].iloc[i] if use_super_longa else 0,
                'EMA_Curta_Slope': df['EMA_Curta_Slope'].iloc[i],
                'EMA_Longa_Slope': df['EMA_Longa_Slope'].iloc[i],
                'EMA_SL_Slope': df['EMA_SL_Slope'].iloc[i] if use_super_longa else 0,
                'Volume': df['Volume'].iloc[i],
                'Volume_EMA_7': df['Volume_EMA_7'].iloc[i],
                'Volume_EMA_30': df['Volume_EMA_30'].iloc[i],
                'Close': close,
                'RSI': rsi,
                'ATR': df['ATR'].iloc[i]
            }
            features_list.append(features)
            targets.append(pct_change)

            if position == 0:
                if use_super_longa:
                    if (df['EMA_Longa'].iloc[i] > df['EMA_Super_Longa'].iloc[i] and
                        df['EMA_Curta'].iloc[i-1] < df['EMA_Curta'].iloc[i] and
                        df['EMA_SL_Slope'].iloc[i] > 0 and rsi < 70):
                        position = 1
                        entry_price = close
                        trades.append((df.index[i], 'BUY', close, 0))
                else:
                    if (df['EMA_Curta'].iloc[i] > df['EMA_Longa'].iloc[i] and
                        df['EMA_Curta_Slope'].iloc[i] > 0 and
                        df['EMA_Longa_Slope'].iloc[i] > 0 and rsi < 70):
                        position = 1
                        entry_price = close
                        trades.append((df.index[i], 'BUY', close, 0))
            elif position == 1:
                if use_super_longa:
                    if (df['EMA_Longa'].iloc[i] < df['EMA_Super_Longa'].iloc[i] or
                        df['EMA_SL_Slope'].iloc[i] < 0 or rsi > 70):
                        position = 0
                        profit = (close - entry_price) * (capital / entry_price)
                        capital += profit
                        trades.append((df.index[i], 'SELL', close, profit))
                else:
                    if (df['EMA_Curta'].iloc[i] < df['EMA_Longa'].iloc[i] or
                        df['EMA_Curta_Slope'].iloc[i] < 0 or rsi > 70):
                        position = 0
                        profit = (close - entry_price) * (capital / entry_price)
                        capital += profit
                        trades.append((df.index[i], 'SELL', close, profit))
        
        if position == 1:
            profit = (df['Close'].iloc[-1] - entry_price) * (capital / entry_price)
            capital += profit
            trades.append((df.index[-1], 'FORCE CLOSE', df['Close'].iloc[-1], profit))

        features_df = pd.DataFrame(features_list)
        targets = targets[:len(features_df)]
        
        if not features_df.empty:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df)
            features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
            self.scalers['1d'] = scaler
            
            X_train, X_test, y_train, y_test = train_test_split(features_df, targets, test_size=0.2, random_state=42)
            self.xgb_model_1d = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
            self.xgb_model_1d.fit(X_train, y_train)
            y_pred = self.xgb_model_1d.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Erro quadrático médio do modelo XGBoost (1d) [teste]: {mse:.6f}")
        else:
            self.xgb_model_1d = None
        
        return capital, trades, df, features_df, targets

    def prepare_data_for_xgb(self, df, interval, look_ahead=5):
        df = self.calculate_indicators(df, interval=interval)
        targets = []
        
        if interval == '1d':
            look_ahead = 5
        elif interval == '1h':
            look_ahead = 12
        else:
            look_ahead = 60
        
        for i in range(len(df) - look_ahead):
            current_price = df['Close'].iloc[i]
            future_price = df['Close'].iloc[i + look_ahead]
            pct_change = (future_price - current_price) / current_price
            targets.append(pct_change)
        
        features_df = df.iloc[:-look_ahead][[
            'EMA_Curta', 'EMA_Longa', 'EMA_Curta_Slope', 'EMA_Longa_Slope',
            'Volume', 'Volume_EMA_7', 'Volume_EMA_30', 'Close', 'RSI', 'ATR',
            'BB_Upper', 'BB_Lower', 'BB_Middle'
        ]].dropna()
        targets = targets[:len(features_df)]
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        features_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
        self.scalers[interval] = scaler
        
        return features_df, targets, scaler

    def train_xgb_model(self, features_df, targets, interval='1h'):
        if features_df.empty or not targets:
            return
        
        # Converter targets para array e remover NaN/infinito
        targets = np.array(targets)
        valid_mask = np.isfinite(targets)
        features_df = features_df.loc[valid_mask]
        targets = targets[valid_mask]
        
        if len(targets) == 0:
            print(f"Erro: Nenhum dado válido para treinar o modelo XGBoost ({interval}).")
            return
        
        X = features_df
        y = targets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if interval == '1d':
            self.xgb_model_1d = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
            self.xgb_model_1d.fit(X_train, y_train)
            y_pred = self.xgb_model_1d.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Erro quadrático médio do modelo XGBoost (1d) [teste]: {mse:.6f}")
        else:
            self.xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
            self.xgb_model.fit(X_train, y_train)
            y_pred = self.xgb_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Erro quadrático médio do modelo XGBoost ({interval}) [teste]: {mse:.6f}")

    def predict_action(self, df, ticker, interval):
        model = self.xgb_model_1d if interval == '1d' else self.xgb_model
        if model is None:
            return "Modelo não treinado", None, None, None, None, None, None
        
        if ticker in self.open_trades:
            entry_price, timestamp, stop_loss, stop_gain, action = self.open_trades[ticker]
            current_price = df['Close'].iloc[-1]
            if stop_loss is not None and stop_gain is not None:
                if action == "Comprar" and (current_price <= stop_loss or current_price >= stop_gain):
                    del self.open_trades[ticker]
                    profit = (current_price - entry_price) if current_price >= stop_gain else (current_price - entry_price)
                    print(f"Posição fechada em {ticker}: {'Lucro' if profit > 0 else 'Prejuízo'} de {profit:.2f}")
                elif action == "Vender" and (current_price >= stop_loss or current_price <= stop_gain):
                    del self.open_trades[ticker]
                    profit = (entry_price - current_price) if current_price <= stop_gain else (entry_price - current_price)
                    print(f"Posição fechada em {ticker}: {'Lucro' if profit > 0 else 'Prejuízo'} de {profit:.2f}")
                else:
                    return f"Posição ativa em {ticker}: {action}", None, stop_loss, stop_gain, None, None, None
        
        if interval == '1d':
            current_features = {
                'EMA_Curta': df['EMA_Curta'].iloc[-1],
                'EMA_Longa': df['EMA_Longa'].iloc[-1],
                'EMA_Super_Longa': df['EMA_Super_Longa'].iloc[-1] if 'EMA_Super_Longa' in df.columns else 0,
                'EMA_Curta_Slope': df['EMA_Curta_Slope'].iloc[-1],
                'EMA_Longa_Slope': df['EMA_Longa_Slope'].iloc[-1],
                'EMA_SL_Slope': df['EMA_SL_Slope'].iloc[-1] if 'EMA_SL_Slope' in df.columns else 0,
                'Volume': df['Volume'].iloc[-1],
                'Volume_EMA_7': df['Volume_EMA_7'].iloc[-1],
                'Volume_EMA_30': df['Volume_EMA_30'].iloc[-1],
                'Close': df['Close'].iloc[-1],
                'RSI': df['RSI'].iloc[-1],
                'ATR': df['ATR'].iloc[-1]
            }
        else:
            current_features = {
                'EMA_Curta': df['EMA_Curta'].iloc[-1],
                'EMA_Longa': df['EMA_Longa'].iloc[-1],
                'EMA_Curta_Slope': df['EMA_Curta_Slope'].iloc[-1],
                'EMA_Longa_Slope': df['EMA_Longa_Slope'].iloc[-1],
                'Volume': df['Volume'].iloc[-1],
                'Volume_EMA_7': df['Volume_EMA_7'].iloc[-1],
                'Volume_EMA_30': df['Volume_EMA_30'].iloc[-1],
                'Close': df['Close'].iloc[-1],
                'RSI': df['RSI'].iloc[-1],
                'ATR': df['ATR'].iloc[-1],
                'BB_Upper': df['BB_Upper'].iloc[-1],
                'BB_Lower': df['BB_Lower'].iloc[-1],
                'BB_Middle': df['BB_Middle'].iloc[-1]
            }
        
        features_df = pd.DataFrame([current_features])
        
        # Normalizar as features
        scaler = self.scalers.get(interval, StandardScaler())
        features_scaled = scaler.fit_transform(features_df) if interval not in self.scalers else scaler.transform(features_df)
        features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
        
        # Calcular ATR normalizado e threshold dinâmico
        atr = df['ATR'].iloc[-1]
        current_price = df['Close'].iloc[-1]
        atr_pct = atr / current_price  # Normalizar ATR pelo preço atual
        threshold = atr_pct * 2  # Threshold dinâmico baseado no ATR
        
        # Calcular stop loss e stop gain em percentuais
        sl_buy = current_price * (1 - atr_pct * 2)  # Stop loss de 2x ATR%
        sg_buy = current_price * (1 + atr_pct * 3)  # Stop gain de 3x ATR%
        sl_sell = current_price * (1 + atr_pct * 2)  # Stop loss de 2x ATR%
        sg_sell = current_price * (1 - atr_pct * 3)  # Stop gain de 3x ATR%
        
        pct_change_pred = model.predict(features_df)[0]
        prob = min(100.0, 50.0 + (abs(pct_change_pred) / threshold) * 50.0)
        
        if pct_change_pred > threshold:
            action = "Comprar"
        elif pct_change_pred < -threshold:
            action = "Vender"
        else:
            action = "Manter"
        
        return action, prob, sl_buy, sg_buy, sl_sell, sg_sell, pct_change_pred

def print_results(initial_capital, final_capital, trades, signal, prob, sl_buy, sg_buy, sl_sell, sg_sell, pct_change):
    print(f"Capital Inicial: ${initial_capital:.2f}")
    print(f"Capital Final: ${final_capital:.2f}")
    if trades:
        profit = final_capital - initial_capital
        print(f"Lucro/Prejuízo: ${profit:.2f}")
        print(f"Retorno: {profit / initial_capital * 100:.2f}%")
        print(f"Total de Trades: {len(trades)}")
        winners = sum(1 for _, _, _, profit in trades if profit > 0)
        losers = len(trades) - winners
        print(f"Trades Vencedores: {winners}")
        print(f"Trades Perdedores: {losers}")
        print("\nÚltimos 5 Trades:")
        for trade in trades[-5:]:
            print(f"{trade[0]} - {trade[1]} @ {trade[2]:.2f} - Lucro: {trade[3]:.2f}")
    print(f"Sinal atual: {signal}")
    print(f"Probabilidade de acerto (XGBoost): {prob:.2f}%")
    print(f"Sugestão para Compra - Stop Loss: {sl_buy:.2f}, Stop Gain: {sg_buy:.2f}")
    print(f"Sugestão para Venda - Stop Loss: {sl_sell:.2f}, Stop Gain: {sg_sell:.2f}")
    print(f"Previsão de Variação: {pct_change * 100:.2f}%")

if __name__ == "__main__":
    ts = TradingStrategy()
    df = ts.get_data("AAPL", period="1y", interval="1d")
    capital, trades, df, _, _ = ts.backtest(df)
    print(f"Capital final: ${capital:.2f}")