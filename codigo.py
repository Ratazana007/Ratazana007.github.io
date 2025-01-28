import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import base64
from io import BytesIO
from flask import Flask, render_template

app = Flask(__name__)

def load_and_process_data():
    # Carregando os dados
    df = pd.read_csv('resultados_selecoes_FIFA.csv')

    def calcular_media_gols(df, time):
        """Calcula a média de gols marcados por uma seleção"""
        gols_mandante = df[df['time_mandante'] == time]['gols_mandante'].sum()
        gols_visitante = df[df['time_visitante'] == time]['gols_visitante'].sum()
        total_gols = gols_mandante + gols_visitante

        jogos_mandante = len(df[df['time_mandante'] == time])
        jogos_visitante = len(df[df['time_visitante'] == time])
        total_jogos = jogos_mandante + jogos_visitante

        return total_gols / total_jogos if total_jogos > 0 else 0

    # Obtendo todas as seleções únicas
    times_unicos = pd.concat([df['time_mandante'], df['time_visitante']]).unique()

    # Calculando média de gols para cada seleção
    medias_gols = {time: calcular_media_gols(df, time) for time in times_unicos}
    medias_df = pd.DataFrame(list(medias_gols.items()),
                            columns=['Seleção', 'Média de Gols por Partida'])

    # Ordenando por média de gols para melhor visualização e removendo médias iguais a zero
    medias_df = medias_df[medias_df['Média de Gols por Partida'] > 0].sort_values('Média de Gols por Partida', ascending=False)

    # Removendo linhas com valores NaN antes de preparar os dados para o modelo
    df_clean = df.dropna(subset=['gols_mandante', 'gols_visitante'])

    # Definindo datas limite para cada período
    data_atual = pd.Timestamp.now()
    limite_15_anos = data_atual - pd.DateOffset(years=15)
    limite_5_anos = data_atual - pd.DateOffset(years=5)
    limite_2_anos = data_atual - pd.DateOffset(years=2)

    # Convertendo coluna de data para datetime
    df_clean['data'] = pd.to_datetime(df_clean['data'])

    # Criando DataFrames filtrados por período
    df_15_anos = df_clean[df_clean['data'] >= limite_15_anos]
    df_5_anos = df_clean[df_clean['data'] >= limite_5_anos]
    df_2_anos = df_clean[df_clean['data'] >= limite_2_anos]

    return df_clean, times_unicos, medias_df, df_15_anos, df_5_anos, df_2_anos

def criar_modelos_por_periodo(df_periodo, times):
    modelos = {}
    for time in times:
        jogos_time = df_periodo[(df_periodo['time_mandante'] == time) |
                               (df_periodo['time_visitante'] == time)]

        if len(jogos_time) > 0:
            X = jogos_time['data'].values.astype(np.int64).reshape(-1, 1)
            y = np.where(jogos_time['time_mandante'] == time,
                        jogos_time['gols_mandante'],
                        jogos_time['gols_visitante']).astype(float)

            modelo = LinearRegression()
            modelo.fit(X, y)
            modelos[time] = modelo
    return modelos

def criar_ranking_previsoes(modelos, periodo, times_unicos):
    data_atual_value = pd.Timestamp.now().value
    previsoes = []

    for selecao in times_unicos:
        if selecao in modelos:
            previsao = max(0, modelos[selecao].predict([[data_atual_value]])[0])
            if previsao > 0 and previsao <= 10:  # Filtrando previsões razoáveis e removendo zeros
                previsoes.append({'Seleção': selecao, 'Gols Previstos': previsao})

    # Criando DataFrame e ordenando
    ranking_df = pd.DataFrame(previsoes)
    ranking_df = ranking_df.sort_values('Gols Previstos', ascending=False)
    ranking_df['Posição'] = range(1, len(ranking_df) + 1)

    # Reordenando colunas
    ranking_df = ranking_df[['Posição', 'Seleção', 'Gols Previstos']]
    
    return ranking_df

def get_plot_url(medias_df):
    plt.figure(figsize=(12, 8))
    plt.hist(medias_df['Média de Gols por Partida'], bins=30, edgecolor='black')
    plt.xlabel('Média de Gols por Partida')
    plt.ylabel('Número de Seleções')
    plt.title('Distribuição da Média de Gols por Seleção')
    plt.grid(True, alpha=0.3)
    
    # Save plot to BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    
    # Convert to base64 string
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/')
def index():
    # Load and process data
    df_clean, times_unicos, medias_df, df_15_anos, df_5_anos, df_2_anos = load_and_process_data()
    
    # Create models
    modelos_15_anos = criar_modelos_por_periodo(df_15_anos, times_unicos)
    modelos_5_anos = criar_modelos_por_periodo(df_5_anos, times_unicos)
    modelos_2_anos = criar_modelos_por_periodo(df_2_anos, times_unicos)
    
    # Create rankings
    ranking_15_anos = criar_ranking_previsoes(modelos_15_anos, 15, times_unicos)
    ranking_5_anos = criar_ranking_previsoes(modelos_5_anos, 5, times_unicos)
    ranking_2_anos = criar_ranking_previsoes(modelos_2_anos, 2, times_unicos)
    
    # Convert rankings to HTML tables with Bootstrap classes
    ranking_15_anos = ranking_15_anos.to_html(classes='table table-striped', index=False)
    ranking_5_anos = ranking_5_anos.to_html(classes='table table-striped', index=False)
    ranking_2_anos = ranking_2_anos.to_html(classes='table table-striped', index=False)
    
    # Get plot
    plot_url = get_plot_url(medias_df)
    
    return render_template('index.html',
                         grafico=plot_url,
                         ranking_15_anos=ranking_15_anos,
                         ranking_5_anos=ranking_5_anos,
                         ranking_2_anos=ranking_2_anos)

if __name__ == '__main__':
    app.run(debug=True)
