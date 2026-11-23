from pathlib import Path
import hashlib
import json
import warnings
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
STOCK_CODE = '20725'
PRODUCT_NAME = 'LUNCH BAG RED RETROSPOT'
SOURCE_URL = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
FREQUENCIES = {'День': 'D', 'Неделя': 'W-SUN', 'Месяц': 'MS'}
HORIZONS = {'День': [1, 7, 31, 61], 'Неделя': [1, 4, 8], 'Месяц': [1, 2]}
MODELS = ['Наивная', 'Эксп. сглаживание', 'ARIMA', 'Бустинг']


def load_data():
    folder = ROOT / 'data/raw'
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'online_retail.zip'
    excel = folder / 'Online Retail.xlsx'
    if not excel.exists():
        if not archive.exists():
            urlretrieve(SOURCE_URL, archive)
        with zipfile.ZipFile(archive) as z:
            excel.write_bytes(z.read('Online Retail.xlsx'))
    df = pd.read_excel(excel, engine='openpyxl')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['StockCode'] = df['StockCode'].astype(str).str.strip()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    digest = hashlib.sha256(excel.read_bytes()).hexdigest()
    return df, digest


def prepare_series(df, stock_code=STOCK_CODE):
    # Считаем проданные штуки без возвратов.
    product = df.loc[df['StockCode'].eq(stock_code)].copy()
    checks = pd.Series({
        'Строк товара': len(product),
        'Отмены': product['InvoiceNo'].str.upper().str.startswith('C').sum(),
        'Количество <= 0': product['Quantity'].le(0).sum(),
        'Цена <= 0': product['UnitPrice'].le(0).sum(),
        'Нет даты': product['InvoiceDate'].isna().sum(),
        'Нет количества': product['Quantity'].isna().sum(),
        'Нет CustomerID': product['CustomerID'].isna().sum(),
        'Полные дубликаты': product.duplicated().sum(),
    }, name='Количество')
    keep = (product['InvoiceDate'].notna() & product['Quantity'].gt(0)
            & product['UnitPrice'].gt(0)
            & ~product['InvoiceNo'].str.upper().str.startswith('C'))
    sales = product.loc[keep].copy()
    # У строк нет ID: одинаковые позиции могут быть разными покупками.
    start = df['InvoiceDate'].min().normalize()
    end = df['InvoiceDate'].max().normalize()
    dates = pd.date_range(start, end, freq='D', name='date')
    daily = sales.groupby(sales['InvoiceDate'].dt.normalize())['Quantity'].sum()
    daily = daily.reindex(dates, fill_value=0).astype(float).rename('sales')
    coverage = pd.DataFrame(index=dates)
    coverage['all_transactions'] = df.groupby(df['InvoiceDate'].dt.normalize()).size().reindex(dates, fill_value=0)
    coverage['product_transactions'] = sales.groupby(sales['InvoiceDate'].dt.normalize()).size().reindex(dates, fill_value=0)
    coverage['sales'] = daily
    coverage['status'] = np.select([
        coverage['all_transactions'].eq(0), coverage['product_transactions'].eq(0)
    ], ['Нет транзакций магазина', 'Нет продаж товара'], default='Есть продажи товара')
    # Последний день выгрузки неполный.
    daily_complete = daily.iloc[:-1]
    weekly = daily_complete.resample('W-SUN').sum()
    week_counts = daily_complete.resample('W-SUN').size()
    weekly = weekly.loc[week_counts.eq(7)].asfreq('W-SUN')
    monthly = daily_complete.resample('MS').sum()
    month_counts = daily_complete.resample('MS').size()
    monthly = monthly.loc[month_counts.to_numpy() == month_counts.index.days_in_month].asfreq('MS')
    series = {'День': daily_complete.asfreq('D'), 'Неделя': weekly, 'Месяц': monthly}
    checks.loc['Строк после фильтрации'] = len(sales)
    return product, sales, checks, coverage, series


def describe_series(series):
    rows = []
    for name, y in series.items():
        rows.append({'Частота': name, 'Начало': y.index.min(), 'Конец': y.index.max(),
                     'Длина': len(y), 'Пропуски': y.isna().sum(), 'Нули': y.eq(0).sum(),
                     'Среднее': y.mean(), 'Медиана': y.median(), 'Станд. отклонение': y.std(),
                     'Минимум': y.min(), 'Максимум': y.max(),
                     'CV': y.std() / y.mean(), 'Доля нулей': y.eq(0).mean()})
    return pd.DataFrame(rows).set_index('Частота')


def outlier_table(y, reference):
    # Порог оцениваем только по началу ряда.
    q1, q3 = reference.quantile([0.25, 0.75])
    threshold = q3 + 1.5 * (q3 - q1)
    return y[y > threshold].rename('sales').to_frame().assign(threshold=threshold)


def stationarity_tests(y, name):
    y = pd.Series(y).dropna()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        adf = adfuller(y, regression='c', autolag='AIC', result_object=False)
        kp = kpss(y, regression='c', nlags='auto', result_object=False)
    return {'Ряд': name, 'n': len(y), 'ADF': adf[0], 'ADF p': adf[1],
            'KPSS': kp[0], 'KPSS p': kp[1],
            'Примечание': ' | '.join(str(w.message) for w in caught)}


def seasonal_diagnostics(y):
    fit = STL(y, period=7, robust=True).fit()
    strength = max(0, 1 - np.var(fit.resid) / np.var(fit.resid + fit.seasonal))
    correlations = acf(y, nlags=28, fft=True)
    f, power = periodogram(y.to_numpy(), detrend='linear')
    return fit, strength, correlations, f, power


def make_features(history, date, frequency):
    lags = {'День': [1, 2, 7, 14], 'Неделя': [1, 2, 4], 'Месяц': [1, 2]}[frequency]
    x = [history[-lag] for lag in lags]
    window = max(lags)
    x += [np.mean(history[-window:]), np.std(history[-window:])]
    if frequency == 'День':
        x += [float(date.day_of_week == d) for d in range(7)]
    # Календарь известен заранее.
    x += [np.sin(2 * np.pi * date.month / 12), np.cos(2 * np.pi * date.month / 12)]
    return x


def forecast_model(train, dates, frequency, model_name):
    h = len(dates)
    values = train.to_numpy(dtype=float)
    if model_name == 'Наивная':
        period = 7 if frequency == 'День' else 1
        prediction = np.resize(values[-period:], h)
    elif model_name == 'Эксп. сглаживание':
        if frequency == 'День':
            model = ExponentialSmoothing(train, trend=None, seasonal='add',
                                         seasonal_periods=7, initialization_method='estimated')
        else:
            model = SimpleExpSmoothing(train, initialization_method='estimated')
        fit = model.fit(optimized=True)
        prediction = fit.forecast(h)
    elif model_name == 'ARIMA':
        fit = ARIMA(train, order=(1, 1, 0), trend='n').fit()
        prediction = fit.forecast(h)
    else:
        lag = {'День': 14, 'Неделя': 4, 'Месяц': 2}[frequency]
        X = [make_features(values[:i], train.index[i], frequency) for i in range(lag, len(train))]
        target = values[lag:]
        model = GradientBoostingRegressor(n_estimators=80, learning_rate=0.05,
                                           max_depth=2, min_samples_leaf=1 if frequency == 'Месяц' else 3,
                                           loss='squared_error', random_state=42)
        model.fit(np.asarray(X), target)
        history = list(values)
        prediction = []
        for date in dates:
            features = make_features(history, date, frequency)
            value = max(0.0, float(model.predict(np.asarray([features]))[0]))
            prediction.append(value)
            # На следующих шагах используем свой прогноз.
            history.append(value)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0)
    assert len(prediction) == h and np.isfinite(prediction).all()
    return prediction


def forecast_metrics(actual, predicted, train, frequency):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = actual - predicted
    period = 7 if frequency == 'День' else 1
    values = np.asarray(train)
    scale = np.abs(values[period:] - values[:-period]).mean()
    if scale <= 0:
        raise ValueError('MASE не определена для постоянного обучающего ряда')
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.divide(200 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
    return {'MAE': np.abs(error).mean(), 'RMSE': np.sqrt(np.mean(error ** 2)),
            'MASE': np.abs(error).mean() / scale, 'sMAPE': smape.mean()}


def run_experiments(series):
    predictions, scores, splits, diagnostics = [], [], [], []
    for frequency, y in series.items():
        max_h = max(HORIZONS[frequency])
        spacing = 7 if frequency == 'День' else 1
        origins = [len(y) - max_h - spacing * k for k in [2, 1, 0]]
        for fold, origin in enumerate(origins, start=1):
            train, test = y.iloc[:origin], y.iloc[origin:origin + max_h]
            assert train.index.max() < test.index.min()
            splits.append({'Частота': frequency, 'Окно': fold, 'Обучение': len(train),
                           'Начало обучения': train.index.min(), 'Конец обучения': train.index.max(),
                           'Начало теста': test.index.min(), 'Конец теста': test.index.max(),
                           'Тест': len(test)})
            for model in MODELS:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    pred = forecast_model(train, test.index, frequency, model)
                for w in caught:
                    diagnostics.append({'Частота': frequency, 'Окно': fold, 'Модель': model,
                                        'Предупреждение': str(w.message)})
                for horizon in HORIZONS[frequency]:
                    metric = forecast_metrics(test.iloc[:horizon], pred[:horizon], train, frequency)
                    scores.append({'Частота': frequency, 'Окно': fold, 'Модель': model,
                                   'Горизонт': horizon, **metric})
                    for step in range(horizon):
                        predictions.append({'Частота': frequency, 'Окно': fold, 'Модель': model,
                                            'Горизонт': horizon, 'Шаг': step + 1,
                                            'Дата прогноза': (train.index.max() + pd.offsets.MonthEnd(0)
                                                              if frequency == 'Месяц' else train.index.max()),
                                            'Дата': test.index[step],
                                            'Факт': test.iloc[step], 'Прогноз': pred[step]})
    return (pd.DataFrame(predictions), pd.DataFrame(scores), pd.DataFrame(splits),
            pd.DataFrame(diagnostics, columns=['Частота', 'Окно', 'Модель', 'Предупреждение']))


def save_series(series):
    paths = {'День': 'daily', 'Неделя': 'weekly', 'Месяц': 'monthly'}
    for frequency, y in series.items():
        y.rename_axis('date').to_csv(ROOT / f'data/processed/{paths[frequency]}.csv')


def load_series():
    paths = {'День': 'daily', 'Неделя': 'weekly', 'Месяц': 'monthly'}
    return {frequency: pd.read_csv(ROOT / f'data/processed/{name}.csv', parse_dates=['date'],
                                  index_col='date')['sales'].asfreq(FREQUENCIES[frequency])
            for frequency, name in paths.items()}
