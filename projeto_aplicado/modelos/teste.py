# Arquivo: debug_preferencias.py
import pandas as pd
import os

print("--- Iniciando Script de Depuração de Preferências ---")

# --- Configuração ---
PASTA_DADOS = 'dados'

try:
    # 1. Carrega os três arquivos de dados essenciais
    df_prof = pd.read_csv(os.path.join(PASTA_DADOS, 'docentes.csv'))
    df_disc = pd.read_csv(os.path.join(PASTA_DADOS, 'disciplinas.csv'))
    df_pref = pd.read_csv(os.path.join(PASTA_DADOS, 'preferencias.csv'))
    print("Arquivos CSV carregados com sucesso.")

    # 2. Verifica se há valores nulos na coluna 'preferencia' do arquivo de entrada
    if df_pref['preferencia'].isnull().any():
        print("\nALERTA: Foram encontrados valores NULOS na coluna 'preferencia' do arquivo 'preferencias.csv'!")
        print("Linhas problemáticas:")
        print(df_pref[df_pref['preferencia'].isnull()])
    else:
        print("\nNenhum valor nulo encontrado na coluna 'preferencia' do arquivo de entrada.")

    # 3. Simula a lógica de preparação da classe Otimizador
    df_prof = df_prof.rename(columns={"id_docente": "id_professor"})
    df_pref = df_pref.rename(columns={"id_docente": "id_professor"})

    lista_professores = df_prof['id_professor'].tolist()
    lista_disciplinas = df_disc['id_disciplina'].tolist()

    # Pivota a tabela
    df_prefs_pivot = df_pref.pivot(
        index='id_professor',
        columns='id_disciplina',
        values='preferencia'
    )
    
    # Preenche quaisquer valores ausentes gerados pelo pivot com 0
    # Esta é a etapa crucial que deve resolver o problema
    df_prefs_pivot_filled = df_prefs_pivot.fillna(0)
    
    # Reindexa para garantir que todas as combinações existam
    df_prefs_final = df_prefs_pivot_filled.reindex(
        index=lista_professores,
        columns=lista_disciplinas,
        fill_value=0 # Garante que novos professores/disciplinas também recebam 0
    )

    # 4. Procura por quaisquer valores NaN restantes na matriz final
    # A variável `stack()` transforma a tabela em uma série, facilitando a busca
    locations = df_prefs_final.stack()[pd.isna(df_prefs_final.stack())]

    if locations.empty:
        print("\nSUCESSO: A matriz de preferências final está limpa e não contém valores NaN.")
        print("O problema provavelmente será resolvido com a correção no 'otimizador_base.py'.")
    else:
        print("\nERRO: Foram encontrados valores NaN na matriz de preferências final, mesmo após o tratamento.")
        print("Combinações problemáticas (Professor, Disciplina):")
        print(locations)

except FileNotFoundError as e:
    print(f"\nERRO: Arquivo não encontrado: {e}")
    print("Verifique se os nomes dos arquivos ('docentes.csv', 'disciplinas.csv', 'preferencias.csv') estão corretos na pasta 'dados/'.")

print("\n--- Fim do Script ---")