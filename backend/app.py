import io # Dosyaları diske kaydetmeden RAM üzerinde işlemek için.
import re
import warnings

# flask> web sunucusunu başlatır
from flask import Flask, render_template, request, flash
# pandas> Veri manipülasyonu ve Excel dosyalarını okumak için
import pandas as pd
# numpy> matematiksel işlemleri yapar
import numpy as np

# standardscaler> verileri standart hale getirir
from sklearn.preprocessing import StandardScaler
# Ridge> lineer model
from sklearn.linear_model import Ridge
# SVR> nonlineer model
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# =============================================================================
# KONTROL PANELİ
# =============================================================================

# --- 1. MODEL & VERİ HASSASİYETİ ---
# Makro ekonominin (kur, faiz) şirket bilançosuna etkisi anında olmaz, gecikmeli (lagged) olur.
MACRO_LAG_YEARS = 3         # generate_future_macros fonksiyonunda makro tahmin yaparken kaç yıllık geçmiş veri kullansın?
MIN_TRAIN_YEARS = 3         # Bir kalemi tahmin etmek için en az kaç yıllık veri olacağı belirlenir.En az 3 yıllık veri yoksa tahmin yapma.
INFLATION_THRESHOLD = 5.0   # Enflasyon/faiz %500'den büyükse (50.0) oran olarak algılanmıştır, 100'e böl

# --- 2. AĞIRLIKLANDIRMA AYARLARI ---
# Örnek: 10 yıllık veride en eski veri 0.5 ağırlık, en yeni veri 1.0 ağırlık alır.
# Böylece model 2023 verisine 2013 verisinden 2x daha fazla önem verir.
WEIGHT_START = 0.5          # 10 yıl önceki ekonomik konjonktür ile bugünkü aynı değil. Eski veriye %50 daha az güven, yeni veriye tam güven (1.0) duyuyoruz.
WEIGHT_END   = 1.0          # En yeni verinin önemi. 1.0 kalmalı

# --- 3. ML HİPERPARAMETRE UZAYI (GRID SEARCH) ---
# Ridge Alpha Listesi (Jüri "daha sert ceza ver" derse buraya ekle)
GRID_RIDGE_ALPHAS = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0]

# SVR Parametreleri
GRID_SVR_C       = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GRID_SVR_EPSILON = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Varsayılan "Güvenli Liman" Değerleri
DEFAULT_RIDGE_ALPHA = 2.0
DEFAULT_SVR_C       = 10.0
DEFAULT_SVR_EPSILON = 0.01

# --- 4. CV FOLD AYARLARI ---
CV_FOLDS_DEFAULT = 3  # Varsayılan geriye dönük test sayısı. Çapraz validasyon sayısı.

# --- 5. EXCEL KOLON EŞLEŞTİRMELERİ (MAPPING) ---
# Sol taraf: Kod içinde kullandığımız Temiz Anahtar
# Sağ taraf: Excel'deki "Ham Başlık" (Value)
EXACT_LABELS_MAP = {
    "Total_Sales":    "Total Sales(Revenues)", 
    "Gross_Profit":   "Gross Profit",
    "ST_Banks_Bonds": "ST Banks / Bonds",
    "LT_Banks_Bonds": "LT Banks / Bonds"
}

# --- 6. HESAPLAMA MANTIĞI GRUPLARI ---
# Buradaki isimler EXACT_LABELS_MAP'in sol tarafındaki KEY'ler ile AYNISI olmalı!
INCOME_STATEMENT_ITEMS = ["Total_Sales", "Gross_Profit"]       # (Enf + Sepet) / 2
BALANCE_SHEET_ITEMS    = ["ST_Banks_Bonds", "LT_Banks_Bonds"] # Sadece Sepet Kur (End)

# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
# =============================================================================

def _to_bytesio(fs):
    data = fs.read()
    return io.BytesIO(data)

def clean_str(val): # Kullanıcı Excel başlığına yanlışlıkla boşluk veya nokta koyarsa kod patlamasın diye 'Regex' ile temizlik.
    if pd.isna(val): return ""
    return re.sub(r"[^a-z0-9]", "", str(val).lower())

def read_macros(path_or_buf):
    # Excel'i okur
    if hasattr(path_or_buf, 'seek'): path_or_buf.seek(0)
    
    try:
        df = pd.read_excel(path_or_buf)
    except Exception:
        raise ValueError("Yüklenen dosya geçerli bir Excel dosyası değil!")

    # Başlıkları standartlaştır (küçük harf, boşluksuz)
    df.columns = [str(c).lower().strip() for c in df.columns] 
    
    df["date"] = pd.to_datetime(df["date"])
    df['year_idx'] = df["date"].dt.year
    grp = df.groupby('year_idx')

    # Yıllık ortalama ve yıl sonu değerlerini hesapla
    df_avg = grp.mean(numeric_only=True).add_suffix('_avg')
    df_end = grp.last(numeric_only=True).add_suffix('_end')

    df = pd.concat([df_avg, df_end], axis=1)
    df.index.name = 'year'

    # Enflasyon ve Faiz Oranı Normalizasyonu (Panelden Kontrol)
    # Excel başlıkları standart olduğu için _avg ve _end eklenmiş hallerini kontrol ediyoruz
    cols_to_check = ["hh_inflation_yoy_avg", "hh_inflation_yoy_end", "tcmb_policy_rate_avg", "tcmb_policy_rate_end"]
    
    for col in cols_to_check:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x / 100.0 if x > INFLATION_THRESHOLD else x)

    df.index = df.index.astype(int)
    return df.sort_index()

def extract_targets_from_financials(file): # Sadece 2000-2050 arası mantıklı yılları alıyoruz. Hatalı tarih girişlerini filtreliyoruz.
    if hasattr(file, 'seek'): file.seek(0)
    df = pd.read_excel(file)

    df = df.set_index(df.columns[0])
    
    valid_years = {}
    for col in df.columns:
        col_str = str(col).replace(".0", "").strip()
        if col_str.isdigit() and 2000 < int(col_str) < 2050:
            valid_years[col] = int(col_str)
            
    df_clean = df[list(valid_years.keys())].rename(columns=valid_years)
    df_clean = df_clean.sort_index(axis=1)
    
    # EXACT_LABELS_MAP kullanarak veriyi çek
    targets = {}
    for key, exact_label in EXACT_LABELS_MAP.items():
        clean_target = clean_str(exact_label)
        found_series = None
        for idx in df_clean.index:
            if clean_str(idx) == clean_target:
                found_series = df_clean.loc[idx]
                break
        
        if found_series is not None:
            s = pd.to_numeric(found_series, errors='coerce')
            s.index = s.index.astype(int)
            targets[key] = s
        else:
            targets[key] = pd.Series(dtype=float)
            
    return targets

# Gelecek yılın makro verisi (Kur, Enflasyon) elimizde yok. Önce ML ile bu makro verileri tahmin ediyoruz.
def generate_future_macros(macros_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    if target_year in macros_df.index: 
        return macros_df

    future_row = {}
    years = sorted(macros_df.index.tolist())
    
    macro_cols = ['usdtry_avg', 'usdtry_end', 'eurtry_avg', 'eurtry_end', 
                  'hh_inflation_yoy_avg', 'hh_inflation_yoy_end',
                  'tcmb_policy_rate_avg', 'tcmb_policy_rate_end']
    
    min_required_macro = MACRO_LAG_YEARS + 1

    for col in macro_cols:
        if col not in macros_df.columns:
            continue
            
        # Yeterli veri yoksa basit ortalama
        if len(years) < min_required_macro:
            avg_growth = macros_df[col].pct_change().mean()
            if pd.isna(avg_growth): avg_growth = 0.0
            future_row[col] = macros_df.iloc[-1][col] * (1 + avg_growth)
            continue
        
        # Ridge ile Makro Tahmini
        X = []
        y = []
        
        for i in range(MACRO_LAG_YEARS, len(years)):
            lags = []
            for k in range(1, MACRO_LAG_YEARS + 1):
                lags.append(macros_df.loc[years[i-k], col])
            lags.append(i) # Time trend
            X.append(lags)
            y.append(macros_df.loc[years[i], col])
 
        # Ridge L2 Regularization: Verideki gürültüyü (noise) ezberlememesi için katsayıları baskılıyoruz (Ceza Mekanizması).
        try:
            model = Ridge(alpha=DEFAULT_RIDGE_ALPHA)
            model.fit(X, y)
            
            future_X_lags = []
            for k in range(1, MACRO_LAG_YEARS + 1):
                future_X_lags.append(macros_df.loc[years[-k], col])
            future_X_lags.append(len(years))
            
            future_row[col] = model.predict([future_X_lags])[0]
            
        except Exception:
            avg_growth = macros_df[col].pct_change().mean()
            if pd.isna(avg_growth): avg_growth = 0.0
            future_row[col] = macros_df.iloc[-1][col] * (1 + avg_growth)

    df_future = pd.DataFrame([future_row], index=[target_year])
    return pd.concat([macros_df, df_future])

# =============================================================================
# 3. TAHMİN MOTORU (SAF LOGARİTMİK + DİNAMİK GRID SEARCH)
# =============================================================================

def forecast_one_ml(y, macros, model_type="ridge_weighted", target_year=None):
    y = y.dropna()
    y.index = y.index.astype(int)
    macros.index = macros.index.astype(int)

    if target_year is None:
        target_year = int(y.index.max()) + 1
    
    # Backtest yaparken geleceği (hedef yılı) modele göstermiyoruz.
    train_end_year = target_year - 1
    y_train = y[y.index <= train_end_year]

    if len(y_train) < MIN_TRAIN_YEARS: 
        if not y.empty: return float(y.iloc[-1]), "N/A"
        return 0.0, "N/A"
    
    base_year = int(y_train.index.max())
    last_val = float(y_train.loc[base_year])
    
    # --- Veri Hazırlığı ---
    macros_extended = generate_future_macros(macros, target_year)
    common_years = y_train.index.intersection(macros_extended.index)
    
    if len(common_years) < 2: return last_val, "Insuff Data"
    
    M = macros_extended.loc[common_years]
    X_pool = pd.DataFrame(index=M.index)
    
    # Modele ham kur yerine "Sepet Kur Artış Hızı" veriyoruz. Burası modelde "Makro Veri exceli" kullandığımız yer.
    # Ridge ve SVR, bu makro veriler ile ciro artışı arasındaki katsayıyı bulur.
    X_pool["basket_avg_growth"] = ((M["usdtry_avg"] + M["eurtry_avg"]) / 2.0).pct_change()
    X_pool["basket_end_growth"] = ((M["usdtry_end"] + M["eurtry_end"]) / 2.0).pct_change()
    X_pool["inflation"] = M["hh_inflation_yoy_avg"]
    X_pool["policy_rate"] = M["tcmb_policy_rate_avg"]
    
    X_pool = X_pool.bfill().ffill().fillna(0)
    
    # Logaritma kullanarak negatif tahminleri engelliyoruz.
    y_train_shifted = y_train.shift(1)
    y_growth = np.log(y_train / y_train_shifted).dropna()
    
    aligned_idx = y_growth.index.intersection(X_pool.index)
    if len(aligned_idx) < MIN_TRAIN_YEARS: return last_val, "Insuff Aligned"
    
    X_train_full = X_pool.loc[aligned_idx]
    Y_train_full = y_growth.loc[aligned_idx]
    
    # --- DİNAMİK GRID SEARCH & CROSS VALIDATION ---
    # Backtest (geriye dönük test) yaparken veri sayısı azalır.
    # Bu yüzden sabit 3 fold yerine verinin yetip yetmediğine bakarız.
    
    n_samples = len(aligned_idx)
    
    # Config'den gelen değeri kullan, ama veri azsa mecburen düşür
    if n_samples >= (CV_FOLDS_DEFAULT * 2):
        cv_folds = CV_FOLDS_DEFAULT  
    elif n_samples >= 4:
        cv_folds = 2  # Veri orta seviyeyse 2 yıl geriye bak
    else:
        cv_folds = 1  # Veri çok azsa (2016-2018 gibi) sadece son yıla bak (Mecburiyet)

    best_avg_error = float('inf')
    best_params = {}

    if "ridge" in model_type:
        param_grid = [{'alpha': a} for a in GRID_RIDGE_ALPHAS]
    elif "svr" in model_type:
        param_grid = [{'C': c, 'epsilon': e} for c in GRID_SVR_C for e in GRID_SVR_EPSILON]
    else:
        param_grid = [{}]
    
    # En iyi parametreyi bulmak için denemeler yap
    for params in param_grid:
        fold_errors = []
        
        # Geriye dönük "Walk-Forward" validation
        for k in range(cv_folds):
            try:
                # Validation yılı: Sondan (k+1). eleman
                val_idx_k = [aligned_idx[-(k+1)]] 
                
                # Eğitim seti: Validation yılından önceki her şey
                train_idx_k = aligned_idx[:-(k+1)]
                
                if len(train_idx_k) < 2: continue # Eğitim için en az 2 veri lazım
                
                X_tr = X_train_full.loc[train_idx_k]
                y_tr = Y_train_full.loc[train_idx_k]
                X_val = X_train_full.loc[val_idx_k]
                y_val = Y_train_full.loc[val_idx_k]
                
                # Normalizasyon
                scaler_cv = StandardScaler()
                X_tr_scaled = scaler_cv.fit_transform(X_tr)
                X_val_scaled = scaler_cv.transform(X_val)
                
                sample_weights_tr = None
                if "weighted" in model_type:
                    sample_weights_tr = np.linspace(WEIGHT_START, WEIGHT_END, len(X_tr))
                
                if "ridge" in model_type:
                    model = Ridge(**params)
                elif "svr" in model_type:
                    model = SVR(kernel='rbf', **params)
                    
                model.fit(X_tr_scaled, y_tr, sample_weight=sample_weights_tr)
                pred_val = model.predict(X_val_scaled)[0]
                
                error = abs(pred_val - float(y_val.iloc[0]))
                fold_errors.append(error)
                
            except:
                continue
        
        if not fold_errors: continue
            
        avg_error = np.mean(fold_errors)
        
        if avg_error < best_avg_error:
            best_avg_error = avg_error
            best_params = params

    # --- Final Model Eğitimi ---
    sample_weights_full = None
    if "weighted" in model_type:
        sample_weights_full = np.linspace(WEIGHT_START, WEIGHT_END, len(X_train_full))

    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_full)

    if "ridge" in model_type:
        if not best_params: best_params = {'alpha': DEFAULT_RIDGE_ALPHA}
        final_model = Ridge(**best_params)
    elif "svr" in model_type:
        if not best_params: best_params = {'C': DEFAULT_SVR_C, 'epsilon': DEFAULT_SVR_EPSILON}
        final_model = SVR(kernel='rbf', **best_params)

    final_model.fit(X_train_final_scaled, Y_train_full, sample_weight=sample_weights_full)

    # --- Gelecek Tahmini ---
    next_macro_row = macros_extended.loc[target_year]
    prev_macro_row = macros_extended.loc[base_year]
    
    future_data = {}
    curr_basket_new = (next_macro_row["usdtry_avg"] + next_macro_row["eurtry_avg"]) / 2.0
    curr_basket_old = (prev_macro_row["usdtry_avg"] + prev_macro_row["eurtry_avg"]) / 2.0
    future_data["basket_avg_growth"] = (curr_basket_new / curr_basket_old) - 1.0

    curr_basket_end_new = (next_macro_row["usdtry_end"] + next_macro_row["eurtry_end"]) / 2.0
    curr_basket_end_old = (prev_macro_row["usdtry_end"] + prev_macro_row["eurtry_end"]) / 2.0
    future_data["basket_end_growth"] = (curr_basket_end_new / curr_basket_end_old) - 1.0

    future_data["inflation"] = next_macro_row["hh_inflation_yoy_avg"]
    future_data["policy_rate"] = next_macro_row["tcmb_policy_rate_avg"]

    future_X = pd.DataFrame([future_data])
    future_X = future_X[X_train_full.columns] # Sütun sırasını garantiye al

    future_X_scaled = scaler_final.transform(future_X)
    pred_val = float(final_model.predict(future_X_scaled)[0])
    
    # Hem parasal sonucu (TL) hem de kullanılan en iyi parametreleri (Grid Search sonucu) döndür.
    return last_val * np.exp(pred_val), best_params
        
# Rule-Based Model: ML kullanmaz. "Kur ne kadar artarsa Ciro da o kadar artar" diyen bakkal hesabı yaklaşımıdır.        
def forecast_one_overlay(y, macros, item_name, target_year=None):
    y = y.dropna()
    if len(y) < 1: return 0.0
    y.index = y.index.astype(int)
    
    if target_year is None: target_year = int(y.index.max()) + 1
    base_year = target_year - 1
    
    if base_year not in y.index: return 0.0
    last_val = float(y.loc[base_year])

    # Gelecek yılın cirosunu tahmin etmek için, önce gelecek yılın makrosunu tahmin ediyoruz.
    macros_extended = generate_future_macros(macros, target_year)
    if target_year not in macros_extended.index: return last_val
    
    next_m = macros_extended.loc[target_year]
    prev_m = macros_extended.loc[base_year]
    
    basket_new_avg = (next_m["usdtry_avg"] + next_m["eurtry_avg"]) / 2.0
    basket_old_avg = (prev_m["usdtry_avg"] + prev_m["eurtry_avg"]) / 2.0
    basket_avg_growth = (basket_new_avg / basket_old_avg) - 1.0

    basket_new_end = (next_m["usdtry_end"] + next_m["eurtry_end"]) / 2.0
    basket_old_end = (prev_m["usdtry_end"] + prev_m["eurtry_end"]) / 2.0
    basket_end_growth = (basket_new_end / basket_old_end) - 1.0

    inf_val = next_m["hh_inflation_yoy_avg"]
    
    # Gelir Tablosu Kalemleri Yıl içindeki ortalama (avg) kurdan etkilenir.
    if item_name in INCOME_STATEMENT_ITEMS:
        growth_rate = (inf_val + basket_avg_growth) / 2.0
    # Bilanço Kalemleri: Yılın son günündeki (Spot) kurdan etkilenir.
    elif item_name in BALANCE_SHEET_ITEMS:
        growth_rate = basket_end_growth
    else:
        growth_rate = inf_val

    return last_val * (1.0 + growth_rate)

def forecast_financials(targets, macros, user_target_year):
    rows = []
    methods = [
        ("ridge_weighted", "AI (Weighted)"),
        ("ridge_standard", "AI (Standard)"),
        ("svr",      "AI (SVR)"),
        ("overlay",     "Macro Overlay")
    ]
    
    target_year = int(user_target_year)
    base_year = target_year - 1 
    
    # OPTİMİZASYON: Makro veriyi döngü dışında 1 kere kesiyoruz.
    # Data Hiding (Veri Gizleme) - Backtest için geleceği görmesin.
    macros_for_training = macros[macros.index <= base_year].copy()
    
    for name, s in targets.items():
        s = s.dropna()
        if s.empty: continue
                
        if base_year not in s.index:
            base_val = 0.0
        else:
            base_val = float(s.loc[base_year])

        real_actual = None
        if target_year in s.index:
            real_actual = float(s.loc[target_year])
        
        # Model Yarıştırma
        for method_key, method_label in methods:
            used_params = "N/A" # Varsayılan değer

            if method_key == "overlay":
                yhat = forecast_one_overlay(s, macros_for_training, name, target_year=target_year)
            else:              
                # ML modelleri artık (tahmin, parametre) döndürüyor
                yhat, used_params = forecast_one_ml(s, macros_for_training, model_type=method_key, target_year=target_year)
                
            delta = yhat - base_val
            p_delta = (delta / base_val * 100.0) if base_val != 0 else 0.0
            
            error_pct = None
            if real_actual is not None and real_actual != 0:
                error_pct = abs(yhat - real_actual) / abs(real_actual) * 100.0
            
            rows.append({
                "Item": name, 
                "BaseYear": base_year, 
                "Actual": base_val,
                "ForecastYear": target_year, 
                "RealActual": real_actual, 
                "Forecast": yhat, 
                "Δ": delta, 
                "%Δ": p_delta,
                "ErrorPct": error_pct,
                "Model": method_label,
                "Params": str(used_params) # Parametreleri yazıya çevirip tabloya ekledik
            })
        
    return pd.DataFrame(rows)

# =============================================================================
# 4. FLASK
# =============================================================================

app = Flask(__name__)
app.secret_key = "dev"

@app.route("/", methods=["GET", "POST"])
def index():
    summary_table = []
    meta = {}
    
    if request.method == "POST":
        try:
            meta["firm"] = request.form.get("firm", "").strip()
            base_year_str = request.form.get("base_year", "2024") 
            user_base_year = int(base_year_str)
            target_year = user_base_year + 1
            
            f_macro = request.files.get("macros")
            f_fin = request.files.get("financials")
            
            if not f_macro or not f_macro.filename:
                flash("Hata: Makro veriler dosyası yüklenmedi!", "danger")
                return render_template("index.html", summary_table=[], meta=meta)
            
            if not f_fin or not f_fin.filename:
                flash("Hata: Finansal tablolar dosyası yüklenmedi!", "danger")
                return render_template("index.html", summary_table=[], meta=meta)

            meta["macro_filename"] = f_macro.filename
            macros = read_macros(_to_bytesio(f_macro))

            meta["fin_filename"] = f_fin.filename
            targets = extract_targets_from_financials(_to_bytesio(f_fin))
                            
            result_df = forecast_financials(targets, macros, target_year)
            
            if not result_df.empty:
                meta["base_year"] = int(result_df.iloc[0]["BaseYear"])
                meta["next_year"] = int(result_df.iloc[0]["ForecastYear"])
                
                unique_items = result_df['Item'].unique()
                
                for item in unique_items:
                    item_df = result_df[result_df['Item'] == item]
                    if item_df.empty: continue
                    
                    base_row = {
                        "Item": item,
                        "Actual": item_df.iloc[0]['Actual'],
                        "RealActual": item_df.iloc[0]['RealActual']
                    }
                    
                    best_error = float('inf')
                    best_model_key = None
                    
                    if base_row['RealActual'] is not None:
                        for _, row in item_df.iterrows():
                            if row['ErrorPct'] is not None and row['ErrorPct'] < best_error:
                                best_error = row['ErrorPct']
                                if "Weighted" in row['Model']: best_model_key = "weighted"
                                elif "Standard" in row['Model']: best_model_key = "standard"
                                elif "SVR" in row['Model']: best_model_key = "svr"
                                else: best_model_key = "overlay"
                    
                    for _, row in item_df.iterrows():
                        if "Weighted" in row['Model']: key_prefix = "weighted"
                        elif "Standard" in row['Model']: key_prefix = "standard"
                        elif "SVR" in row['Model']: key_prefix = "svr"
                        else: key_prefix = "overlay"
                            
                        base_row[f"{key_prefix}_forecast"] = row['Forecast']
                        base_row[f"{key_prefix}_pct"] = row['%Δ']
                        base_row[f"{key_prefix}_error"] = row['ErrorPct']
                        base_row[f"{key_prefix}_is_winner"] = (key_prefix == best_model_key)
                        base_row[f"{key_prefix}_params"] = row['Params'] 
                        
                    summary_table.append(base_row)

        except Exception as e:
            flash(f"Hata: {str(e)}", "danger")

    return render_template("index.html", summary_table=summary_table, meta=meta)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)