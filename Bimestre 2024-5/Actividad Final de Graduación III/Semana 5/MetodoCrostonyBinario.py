

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Para modelos binarios y Croston:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_absolute_error
from darts import TimeSeries
from darts.models import Croston


# 1. LECTURA DE DATOS

data_path = 'db.csv'  #Ruta
df = pd.read_csv(f'{data_path}', sep=',')

print(df.info())
display(df.head(10))


# 2. PREPROCESAMIENTO INICIAL


def process_date_str(number):
    """
    Convierte un int tipo ddmmaaaa (ej: 10052022) a 'yyyy-mm-dd'.
    """
    string = str(number)
    if len(string) == 7:
        # Caso en que falta un dígito
        string = '0' + string
    # Extrae (dd, mm, aaaa) => ([:2], [2:4], [4:])
    # y formatea a 'aaaa-mm-dd'
    return f'{string[4:]}-{string[2:4]}-{string[:2]}'

df['time_str'] = df['Billing date'].apply(process_date_str)
df['time'] = pd.to_datetime(df['time_str'], format='%Y-%m-%d')

# Se ajusta el lunes como dia inicial (opcional)
df['time'] = df['time'] - pd.to_timedelta(df['time'].dt.weekday, unit='D')

# Limpiamos Net price (reemplazando comas por puntos)
df['Net price'] = df['Net price'].apply(lambda x: x.replace(',', '.')).astype(float)


# 3. AGRUPACIÓN DIARIA + PREPARACIÓN

grouped_df = df.groupby(['Material', 'time']).sum().reset_index()[
    ['time', 'Material', 'Phisical quantity', 'Net price']
]
grouped_df = grouped_df.rename(columns={
    'Material': 'item',
    'Phisical quantity': 'sales',
    'Net price': 'price'
})
grouped_df = grouped_df.sort_values(by='time')


def fill_no_sales_period(df, item_id):
    """
    Rellena con filas (sales=0) aquellos días que no existen en df para el item_id.
    """
    item_df = df[df['item'] == item_id]
    existing_dates = item_df['time'].to_list()
    if len(item_df) == 0:
        return df  # Si no hay datos de ese item, retornamos como está

    first_sale_date = item_df['time'].min().strftime("%Y-%m-%d")
    last_sale_date = item_df['time'].max().strftime("%Y-%m-%d")

    # Frecuencia diaria ('D'). 
    time_index = pd.date_range(start=first_sale_date, end=last_sale_date, freq='D')

    new_data = {'time': [], 'item': [], 'sales': [], 'price': []}
    for date in time_index:
        if date not in existing_dates:
            new_data['time'].append(date)
            new_data['item'].append(item_id)
            new_data['sales'].append(0)
            new_data['price'].append(0.0)

    new_dates_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_dates_df], axis=0)
    return df

items = list(grouped_df['item'].unique())
full_daily_df = grouped_df.copy()

for item in items:
    full_daily_df = fill_no_sales_period(full_daily_df, item)

full_daily_df['sales'] = full_daily_df['sales'].astype(int)
full_daily_df['item'] = full_daily_df['item'].astype(int)


# 4. CLASIFICACIÓN DE DEMANDA (p y CV2)

def calculate_p_interval(df, item):
    filtered_df = df[df['item'] == item]
    total_periods = float(len(filtered_df))
    non_zero_demands = float(len(filtered_df[filtered_df['sales'] > 0]))
    if non_zero_demands == 0:
        return np.inf
    return total_periods / non_zero_demands

def calculate_CV2(df, item):
    filtered_df = df[(df['item'] == item) & (df['sales'] > 0)]
    if len(filtered_df) < 2:
        return 0
    std = np.std(filtered_df['sales'])
    mean_ = np.mean(filtered_df['sales'])
    if mean_ == 0:
        return 0
    return (std / mean_) ** 2

demand_factors = []
for it in items:
    p_val = calculate_p_interval(full_daily_df, it)
    cv2_val = calculate_CV2(full_daily_df, it)
    demand_factors.append((it, p_val, cv2_val))

classification_df = pd.DataFrame(demand_factors, columns=['item', 'p', 'cv2'])

def classify_demand(row):
    THRESHOLD_P = 1.32
    THRESHOLD_CV2 = 0.49
    p = row['p']
    cv2 = row['cv2']
    if p < THRESHOLD_P and cv2 >= THRESHOLD_CV2:
        return 'erratic'
    elif p >= THRESHOLD_P and cv2 >= THRESHOLD_CV2:
        return 'lumpy'
    elif p < THRESHOLD_P and cv2 < THRESHOLD_CV2:
        return 'smooth'
    else:
        return 'intermittent'

classification_df['demand_type'] = classification_df.apply(classify_demand, axis=1)

# Precio unitario promedio
full_daily_df['price'] = full_daily_df['price'].fillna(0.0)
full_daily_df['unit_price'] = np.where(
    full_daily_df['sales'] > 0,
    full_daily_df['price'] / full_daily_df['sales'],
    0
)
filtered_nonzero = full_daily_df[full_daily_df.sales > 0]
avg_price_df = filtered_nonzero[['item', 'unit_price']].groupby('item').mean().reset_index()
avg_price_df.rename(columns={'unit_price': 'avg_unit_price'}, inplace=True)

item_df = classification_df.merge(avg_price_df, on='item', how='left')


# 5. ENTRENAMIENTO Y COMPARACIÓN (TODOS LOS ÍTEMS)

results_list = []
all_daily_preds_list = []  # Para luego generar predicciones mensuales

item_ids = classification_df['item'].unique()

for it in item_ids:
    item_data = full_daily_df[full_daily_df.item == it].sort_values('time').reset_index(drop=True)

    # Si hay pocos datos, saltamos
    if len(item_data) < 10:
        results_list.append({
            'item': it,
            'demand_type': classification_df.loc[classification_df['item']==it, 'demand_type'].values[0],
            'n_points': len(item_data),
            'bin_reg_mae': None,
            'croston_mae': None,
            'croston_est_mae': None
        })
        continue


    # (A) MODELO BINARIO + REGRESIÓN

    item_data['has_sales'] = (item_data['sales'] > 0).astype(int)
    item_data['month_idx'] = item_data['time'].dt.month
    item_data['dow'] = item_data['time'].dt.weekday  # Podrías usarlo si quisieras
    item_data['lag1'] = item_data['sales'].shift(1).fillna(0)
    item_data['lag2'] = item_data['sales'].shift(2).fillna(0)

    train_size = int(len(item_data)*0.8)
    train_df = item_data.iloc[:train_size].copy()
    test_df  = item_data.iloc[train_size:].copy()

    if len(train_df)==0 or len(test_df)==0:
        results_list.append({
            'item': it,
            'demand_type': classification_df.loc[classification_df['item']==it, 'demand_type'].values[0],
            'n_points': len(item_data),
            'bin_reg_mae': None,
            'croston_mae': None,
            'croston_est_mae': None
        })
        continue

    X_train = train_df[['month_idx', 'lag1', 'lag2']]  # Ejemplo: mes del año + lags
    y_train = train_df['has_sales']
    X_test  = test_df[['month_idx', 'lag1', 'lag2']]
    y_test  = test_df['has_sales']

    clf = LogisticRegression()
    try:
        clf.fit(X_train, y_train)
    except:
        results_list.append({
            'item': it,
            'demand_type': classification_df.loc[classification_df['item']==it, 'demand_type'].values[0],
            'n_points': len(item_data),
            'bin_reg_mae': None,
            'croston_mae': None,
            'croston_est_mae': None
        })
        continue

    # Predicción binaria
    test_df['pred_has_sales'] = clf.predict(X_test)
    test_df['pred_sales_bin'] = 0.0

    # Regresión para la magnitud
    reg = LinearRegression()
    train_df_reg = train_df[train_df['sales']>0].copy()
    bin_reg_mae = None
    if len(train_df_reg)>0:
        X_train_reg = train_df_reg[['month_idx', 'lag1', 'lag2']]
        y_train_reg = train_df_reg['sales']

        try:
            reg.fit(X_train_reg, y_train_reg)
        except:
            pass

        idx_has_sales = test_df.index[test_df['pred_has_sales']==1]
        X_test_reg = test_df.loc[idx_has_sales, ['month_idx', 'lag1', 'lag2']]
        if not X_test_reg.empty:
            y_pred_reg = reg.predict(X_test_reg)
            test_df.loc[idx_has_sales, 'pred_sales_bin'] = y_pred_reg

        bin_reg_mae = mean_absolute_error(test_df['sales'], test_df['pred_sales_bin'])


    # (B) CROSTON PURO

    item_ts = TimeSeries.from_dataframe(
        item_data, time_col='time', value_cols='sales', freq='D'
    )
    train_ts = item_ts[:train_size]
    test_ts  = item_ts[train_size:]

    croston_model = Croston()
    try:
        croston_model.fit(train_ts)
        croston_pred = croston_model.predict(len(test_ts))
        croston_values = croston_pred.values().flatten()
        test_values = test_ts.values().flatten()
        croston_mae_val = np.mean(np.abs(test_values - croston_values))
    except:
        croston_values = [np.nan]*len(test_df)
        croston_mae_val = None

    # Pasar predicciones de Croston a DataFrame para merge
    croston_pred_df = pd.DataFrame({
        'time': test_df['time'],
        'pred_croston': croston_values
    })


    # (C) CROSTON + ESTACIONALIDAD (MES del AÑO)

    # 1. Calculamos factor estacional por mes del año
    #    (ejemplo para tomar: moy=1..12 => factor = mean(sales en ese mes) / mean(sales global))
    item_data['moy'] = item_data['time'].dt.month
    global_mean = item_data['sales'].mean() if item_data['sales'].mean() != 0 else 1e-6
    season_factor = item_data.groupby('moy')['sales'].mean() / global_mean

    def apply_seasonal_adjustment(row):
        if row['sales'] == 0:
            return 0
        return row['sales'] / season_factor.loc[row['moy']]

    item_data['sales_adj'] = item_data.apply(apply_seasonal_adjustment, axis=1)

    # 2. Entrenar Croston sobre 'sales_adj'
    item_ts_adj = TimeSeries.from_dataframe(
        item_data, time_col='time', value_cols='sales_adj', freq='D'
    )
    train_ts_adj = item_ts_adj[:train_size]
    test_ts_adj  = item_ts_adj[train_size:]

    croston_model_adj = Croston()
    try:
        croston_model_adj.fit(train_ts_adj)
        croston_pred_adj = croston_model_adj.predict(len(test_ts_adj))
        adj_values = croston_pred_adj.values().flatten()

        # Revertir la estacionalidad
        test_df_adj = item_data.iloc[train_size:].copy()
        test_df_adj['pred_croston_est'] = adj_values

        def revert_season_factor(row):
            return row['pred_croston_est'] * season_factor.loc[row['moy']]

        test_df_adj['pred_croston_est'] = test_df_adj.apply(revert_season_factor, axis=1)

        croston_est_mae_val = mean_absolute_error(
            test_df_adj['sales'],
            test_df_adj['pred_croston_est']
        )
    except:
        croston_est_mae_val = None
        test_df_adj = test_df.copy()
        test_df_adj['pred_croston_est'] = np.nan

    # DataFrame con predicción "estacional" para merge
    croston_est_pred_df = test_df_adj[['time', 'pred_croston_est']].copy()


    # (D) Unimos las predicciones diarias (test set)
    #     - Real, binario, croston, croston_est

    test_pred_df = test_df[['time','sales','pred_sales_bin']].merge(
        croston_pred_df, on='time', how='left'
    ).merge(
        croston_est_pred_df, on='time', how='left'
    )
    test_pred_df['item'] = it

    all_daily_preds_list.append(test_pred_df)


    # (E) Guardamos las métricas globales

    results_list.append({
        'item': it,
        'demand_type': classification_df.loc[classification_df['item']==it, 'demand_type'].values[0],
        'n_points': len(item_data),
        'bin_reg_mae': bin_reg_mae,
        'croston_mae': croston_mae_val,
        'croston_est_mae': croston_est_mae_val
    })


# 6. METRICAS FINALES POR ÍTEM

results_df = pd.DataFrame(results_list)
results_df.sort_values(by='item', inplace=True)
results_df.reset_index(drop=True, inplace=True)

display(results_df.head(50))
# results_df.to_csv('comparacion_croston_mes_ano.csv', sep=';', index=False)


# 7. GENERAR Y EXPORTAR PREDICCIONES MENSUALES

all_daily_preds_df = pd.concat(all_daily_preds_list, ignore_index=True)
all_daily_preds_df['time'] = pd.to_datetime(all_daily_preds_df['time'])
all_daily_preds_df.set_index('time', inplace=True)

# Agrupamos por (item, mes) sumando
monthly_preds = (
    all_daily_preds_df
    .groupby(['item', pd.Grouper(freq='M')])
    .agg({
        'sales': 'sum',
        'pred_sales_bin': 'sum',
        'pred_croston': 'sum',
        'pred_croston_est': 'sum'
    })
    .reset_index()
    .rename(columns={
        'time': 'month',
        'sales': 'y_real',
        'pred_sales_bin': 'y_binario',
        'pred_croston': 'y_croston',
        'pred_croston_est': 'y_croston_est'
    })
    .sort_values(by=['item','month'])
    .reset_index(drop=True)
)

display(monthly_preds.head(50))
monthly_preds.to_excel('predicciones_mensuales_croston_est_mes.xlsx', index=False)
print("Archivo 'predicciones_mensuales_croston_est_mes.xlsx' generado con éxito.")
