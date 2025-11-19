# config.py
import os

# --- Caminhos Base ---
PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(PASTA_RAIZ, 'dados')

# --- Arquivos de Dados Padrão ---
CONFIG_BASE_ARQUIVOS = {
    "disciplinas": os.path.join(PASTA_DADOS, "disciplinas.csv"),
    "professores": os.path.join(PASTA_DADOS, "docentes.csv"),
    "preferencias": os.path.join(PASTA_DADOS, "preferencias.csv"),
    "conflitos": os.path.join(PASTA_DADOS, "matriz_conflitos.csv"),
}

# --- Configurações Padrão por Algoritmo ---

# Programação Linear Inteira (PLI)
CONFIG_PLI_PADRAO = {
    "ARQUIVOS_DADOS": CONFIG_BASE_ARQUIVOS,
    "PENALIDADE_W": 4.0,           # Peso da penalidade para preferências baixas
    "ALOCACOES_FIXAS": [],       # Lista de alocações pré-definidas
}

# Algoritmo de Colônia de Formigas (ACO)
CONFIG_ACO_PADRAO = {
    "ARQUIVOS_DADOS": CONFIG_BASE_ARQUIVOS,
    "ACO_PARAMS": {
        "n_formigas": 20,          # Número de formigas por geração
        "n_geracoes": 150,         # Número de gerações (iterações)
        "alfa": 1.0,               # Importância do feromônio
        "beta": 2.0,               # Importância da heurística (preferência)
        "taxa_evaporacao": 0.2,    # Taxa de evaporação (0 a 1)
    },
    "ALOCACOES_FIXAS": [],
}

# Algoritmo Genético (AG)
CONFIG_AG_PADRAO = {
    "ARQUIVOS_DADOS": CONFIG_BASE_ARQUIVOS,
    "AG_PARAMS": {
        "n_populacao": 100,        # Tamanho da população
        "n_geracoes": 300,         # Número de gerações
        "taxa_crossover": 0.85,    # Probabilidade de crossover (0 a 1)
        "taxa_mutacao": 0.05,      # Probabilidade de mutação por gene (0 a 1)
        "tamanho_torneio": 5,      # Número de indivíduos na seleção por torneio
        "fator_penalidade": 1000,  # Grandeza da penalidade (carga horária, conflitos)
    },
    "ALOCACOES_FIXAS": [],
}

