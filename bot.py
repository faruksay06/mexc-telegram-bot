import time
import requests
import pandas as pd
import numpy as np
import ccxt

# ============================== SİZİN BİLGİLERİNİZ ==============================

TELEGRAM_BOT_TOKEN = "8659912569:AAEqz8BLTa6KV5mRWIdsqThQCBgTI-zo4HQ"
TELEGRAM_CHAT_ID = "8586215391"

SYMBOL = "XRP/USDT"                  # MEXC üzerindeki XRP/USDT işlem çifti
TIMEFRAMES = ["1m", "5m", "15m"]     # MEXC tarafından desteklenen zaman dilimleri
POLL_INTERVAL = 5                    # Döngü kontrol aralığı (saniye)

# ============================== İNDİKATÖR VE SİNYAL PARAMETRELERİ ==============================

# --- Sinyal & Filtre Parametreleri ---
MIN_PCT_REQ = 0.25                   # Min. Pivot Yüzde Değişimi (%)
MIN_DIST = 5                         # İki Sinyal Arası Min Mum Sayısı
PIV_LEN = 10                         # Pivot Periyodu
USE_BODY = True                      # Pivotlar İçin Gövde Kullanımı (False: Fitil, True: Gövde)

# --- SMI Osilatör Parametreleri ---
OSC_LEN = 10                         # SMI Periyodu
SMI_SMOOTH = 2                       # SMI Yumuşatma Ayarı
SMI_OB = 50.0                        # SMI - Aşırı Alım (OB) Eşiği
SMI_OS = -50.0                       # SMI - Aşırı Satım (OS) Eşiği


# ============================== YARDIMCI FONKSİYONLAR ==============================

def send_telegram_message(message: str):
    """Telegram üzerinden bildirim gönderir."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        print(f"❌ Telegram Mesaj Gönderme Hatası: {e}")
        return None


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Pine Script ta.ema hesaplaması ile birebir uyumlu EMA."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_smi(df: pd.DataFrame, osc_len: int, smi_smooth: int) -> pd.Series:
    """
    Pine Script SMI Hesaplaması:
    hh = ta.highest(high, osc_len)
    ll = ta.lowest(low, osc_len)
    diff = hh - ll
    rdiff = close - (hh + ll) / 2
    avgrel = ta.ema(ta.ema(rdiff, smi_smooth), smi_smooth)
    avgdiff = ta.ema(ta.ema(diff, smi_smooth), smi_smooth)
    smi_val = avgdiff != 0 ? (avgrel / (avgdiff / 2)) * 100 : 0.0
    """
    hh = df['high'].rolling(window=osc_len).max()
    ll = df['low'].rolling(window=osc_len).min()
    diff = hh - ll
    rdiff = df['close'] - (hh + ll) / 2.0

    avgrel = calculate_ema(calculate_ema(rdiff, smi_smooth), smi_smooth)
    avgdiff = calculate_ema(calculate_ema(diff, smi_smooth), smi_smooth)

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2.0)) * 100.0, 0.0)
    return pd.Series(smi, index=df.index)


def calculate_pivots_and_signals(df: pd.DataFrame):
    """
    Pivot Noktaları, SMI Kesişimleri ve Sinyal Koşullarını Hesaplar.
    """
    n = len(df)
    
    # 1. Pivot Kaynak Fiyatlarının Seçimi
    if USE_BODY:
        src_high = np.maximum(df['open'].values, df['close'].values)
        src_low = np.minimum(df['open'].values, df['close'].values)
    else:
        src_high = df['high'].values
        src_low = df['low'].values

    # 2. Pivot High / Low Hesaplaması (ta.pivothigh / ta.pivotlow)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)

    for i in range(PIV_LEN, n - PIV_LEN):
        # Pivot High
        window_h = src_high[i - PIV_LEN : i + PIV_LEN + 1]
        if src_high[i] == np.max(window_h):
            ph[i] = src_high[i]

        # Pivot Low
        window_l = src_low[i - PIV_LEN : i + PIV_LEN + 1]
        if src_low[i] == np.min(window_l):
            pl[i] = src_low[i]

    df['ph'] = ph
    df['pl'] = pl

    # 3. Son Pivot Fiyatının İzlenmesi (last_piv_price)
    last_piv_price = np.full(n, np.nan)
    curr_piv = np.nan

    for i in range(n):
        if not np.isnan(pl[i]):
            curr_piv = pl[i]
        if not np.isnan(ph[i]):
            curr_piv = ph[i]
        last_piv_price[i] = curr_piv

    df['last_piv_price'] = last_piv_price

    # 4. SMI Hesaplaması
    df['smi'] = calculate_smi(df, OSC_LEN, SMI_SMOOTH)

    # 5. Yüzde Değişim (raw_pct_change ve abs_pct_change)
    df['raw_pct_change'] = np.where(
        ~np.isnan(df['last_piv_price']) & (df['last_piv_price'] != 0),
        ((df['close'] - df['last_piv_price']) / df['last_piv_price']) * 100.0,
        np.nan
    )
    df['abs_pct_change'] = np.abs(df['raw_pct_change'])

    # 6. Kesişimler (ta.crossover & ta.crossunder)
    smi_prev = df['smi'].shift(1)
    df['crossover_os'] = (smi_prev < SMI_OS) & (df['smi'] >= SMI_OS)
    df['crossunder_ob'] = (smi_prev > SMI_OB) & (df['smi'] <= SMI_OB)

    # 7. Sinyal Mantığı ve İki Sinyal Arası Min Mum Kontrolü
    df['buy_cond'] = False
    df['sell_cond'] = False

    last_signal_bar = -99999

    for i in range(n):
        piv_pct_ok = not np.isnan(df['abs_pct_change'].iloc[i]) and df['abs_pct_change'].iloc[i] >= MIN_PCT_REQ
        can_signal_dist = (i - last_signal_bar) >= MIN_DIST

        if df['crossover_os'].iloc[i] and piv_pct_ok and can_signal_dist:
            df.iat[i, df.columns.get_loc('buy_cond')] = True
            last_signal_bar = i

        elif df['crossunder_ob'].iloc[i] and piv_pct_ok and can_signal_dist:
            df.iat[i, df.columns.get_loc('sell_cond')] = True
            last_signal_bar = i

    return df


# ============================== ANA ÇALIŞMA DÖNGÜSÜ ==============================

def main():
    # MEXC Borsa Kurulumu
    exchange = ccxt.mexc({'enableRateLimit': True})
    print(f"🚀 Çoklu Zaman Dilimli Bot Başlatıldı!")
    print(f"📌 Borsa: MEXC | Sembol: {SYMBOL} | Zaman Dilimleri: {TIMEFRAMES}")
    print("--------------------------------------------------")

    # Her zaman diliminin son işlenen mum zamanını bağımsız takip etmek için sözlük
    last_processed_timestamps = {tf: None for tf in TIMEFRAMES}

    while True:
        try:
            # Zaman dilimlerini sırayla kontrol et
            for tf in TIMEFRAMES:
                ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=200)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Kapanmamış canlı mumu hariç tutuyoruz
                completed_df = df.iloc[:-1].copy()
                completed_df = calculate_pivots_and_signals(completed_df)

                last_row = completed_df.iloc[-1]
                last_bar_time = last_row['timestamp']

                # İlgili zaman diliminde yeni bir mum kapandıysa kontrol et
                if last_processed_timestamps[tf] != last_bar_time:
                    is_buy = last_row['buy_cond']
                    is_sell = last_row['sell_cond']

                    if is_buy or is_sell:
                        signal_type = "LONG 🟢" if is_buy else "SHORT 🔴"
                        close_price = last_row['close']
                        smi_val = round(last_row['smi'], 2)
                        piv_pct = round(last_row['raw_pct_change'], 2)

                        msg = (
                            f"⚠️ *MESUT - ULTRA v3.26 SİNYALİ*\n\n"
                            f"🔹 *Sembol:* `{SYMBOL}` (MEXC)\n"
                            f"🔹 *Yön:* {signal_type}\n"
                            f"🔹 *Giriş Fiyatı:* `{close_price}`\n"
                            f"🔹 *SMI Değeri:* `{smi_val}`\n"
                            f"🔹 *Pivot Değişimi:* `%{piv_pct}`\n"
                            f"⏱ *Zaman Dilimi:* `{tf}`"
                        )

                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚨 [{tf}] Sinyal Yakalandı: {signal_type} | Fiyat: {close_price}")
                        send_telegram_message(msg)

                    last_processed_timestamps[tf] = last_bar_time

        except Exception as e:
            print(f"⚠️ Hata Oluştu: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
