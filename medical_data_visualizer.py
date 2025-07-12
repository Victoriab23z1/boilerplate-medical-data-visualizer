import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar los datos
df = pd.read_csv('medical_examination.csv')

# 2. Añadir columna 'overweight'
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['overweight'] > 25).astype(int)

# 3. Normalizar datos: 0 siempre es bueno, 1 siempre es malo
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Gráfico categórico
def draw_cat_plot():
    # 5. Crear df_cat
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Agrupar y reformatear
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})

    # 7. Crear gráfico
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio',
                      data=df_cat, kind='bar').fig

    # 9. Guardar y retornar
    fig.savefig('catplot.png')
    return fig

# 10. Mapa de calor
def draw_heat_map():
    # 11. Limpiar datos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Matriz de correlación
    corr = df_heat.corr()

    # 13. Crear máscara para la triangular superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar figura
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Graficar heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})

    # 16. Guardar y retornar
    fig.savefig('heatmap.png')
    return fig

