import sys, os
import pandas as pd

# Ajusta caminho para importação local
caminho_do_projeto = r'C:\Users\kmenezes\OneDrive - unb.br\tcc_organizacao'
if caminho_do_projeto not in sys.path:
    sys.path.insert(0, caminho_do_projeto)

from projeto_aplicado.modelos.otimizador_aco_instrumentado import OtimizadorACOInstrumentado

CONFIG = {
    "ARQUIVOS_DADOS": {
        "disciplinas": "disciplinas.csv",
        "professores": "docentes.csv",
        "preferencias": "preferencias.csv",
        "conflitos": "matriz_conflitos.csv"
    },
    "ACO_PARAMS": {
        "n_formigas": 5,
        "n_geracoes": 3,
        "alfa": 1.0,
        "beta": 2.0,
        "taxa_evaporacao": 0.2
    }
}

if __name__ == "__main__":
    otm = OtimizadorACOInstrumentado(CONFIG)
    resultado = otm.resolver()
    print("Chaves resultado:", list(resultado.keys()))
    print("Metricas (geracao, melhor_global):", [(m['geracao'], m['melhor_global']) for m in resultado['metricas_iteracao']])
    print("Snapshots feromônio:", len(resultado['pheromone_snapshots']))
    print("Shape snapshot 0:", resultado['pheromone_snapshots'][0].shape)
    print("Long DF linhas:", len(resultado['pheromone_long_df']))
    print("Eventos melhor global:", resultado['eventos_melhor_global'])
