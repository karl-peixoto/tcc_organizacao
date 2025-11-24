"""Script de geração de amostras para estudos com ACO Fast.

Permite dois modos de execução usando o mesmo arquivo:
    1. Estudo de Convergência (modo 'convergencia'):
         - Gera N execuções aleatórias variando hiperparâmetros dentro de domínios discretos/intervalos.
         - Salva dois arquivos CSV: resumo e histórico das métricas por geração.
    2. Estudo de Sensibilidade via Latin Hypercube (modo 'lhs'):
         - Amostra N combinações de hiperparâmetros contínuos usando LatinHypercube.
         - Salva dois arquivos CSV: resumo e histórico das métricas por geração.

Exemplos de uso (cmd.exe):
    python simular_acos.py --modo convergencia --n-execs 12 --geracoes 150 --saida resultados_analises_simulacoes
    python simular_acos.py --modo lhs --n-amostras 120 --geracoes 100 --saida resultados_analises_simulacoes

Parâmetros principais:
    --modo            Tipo de estudo: 'convergencia' ou 'lhs'
    --n-execs         Número de execuções (apenas convergencia)
    --n-amostras      Número de amostras (apenas lhs)
    --geracoes        Número de gerações por execução
    --seed            Seed para reprodutibilidade
    --saida           Diretório base onde salvar subpastas e CSVs

Saídas criadas:
    <saida>/aco_convergencia/aco_convergencia_resumo.csv
    <saida>/aco_convergencia/aco_convergencia_historico.csv
    <saida>/aco_lhs/aco_lhs_resumo.csv
    <saida>/aco_lhs/aco_lhs_historico.csv

O histórico inclui colunas de métricas por geração produzidas pelo OtimizadorACOFast.
"""

import argparse
from scipy.stats import qmc
import pandas as pd
import numpy as np
import time
import os
import sys
import random



caminho_do_projeto = r'C:\Users\kmenezes\OneDrive - unb.br\tcc_organizacao'

if caminho_do_projeto not in sys.path:
    sys.path.insert(0, caminho_do_projeto)

from projeto_aplicado.modelos.otimizador_base import Otimizador
from projeto_aplicado.modelos.analisador import AnalisadorDeSolucao
from projeto_aplicado.modelos.otimizador_aco_fast import OtimizadorACOFast
from projeto_aplicado.modelos.otimizador_aco import OtimizadorACO


# Configurações de visualização

CONFIG_BASE = {
    "ARQUIVOS_DADOS": {
        "disciplinas": "disciplinas.csv",
        "professores": "docentes.csv",
        "preferencias": "preferencias.csv",
        "conflitos": "matriz_conflitos.csv" 
    }
}

def inicializar_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def executar_single(params_aco: dict) -> dict:
    """Executa uma única execução do ACOFast e retorna dict com resultado e histórico."""
    config_run = {**CONFIG_BASE, "ACO_PARAMS": params_aco}
    t0 = time.time()
    otm = OtimizadorACOFast(config=config_run)
    res = otm.resolver()
    tempo = time.time() - t0
    hist = pd.DataFrame(res.get("metricas_iteracao", []))
    return {
        "resultado": res,
        "historico": hist,
        "tempo": tempo
    }


def estudo_convergencia(n_execucoes: int, n_geracoes: int, seed: int) -> tuple:
    """Executa estudo de convergência com variação aleatória de hiperparâmetros."""
    inicializar_seed(seed)
    DOM_FORMIGAS = [10, 15, 20, 30, 40, 50]
    DOM_ALFA = (0.6, 2.0)
    DOM_BETA = (1.0, 6.0)
    DOM_RHO = (0.05, 0.5)

    resultados = []
    historicos = []
    for exec_id in range(n_execucoes):
        params = {
            "n_formigas": random.choice(DOM_FORMIGAS),
            "n_geracoes": n_geracoes,
            "alfa": random.uniform(*DOM_ALFA),
            "beta": random.uniform(*DOM_BETA),
            "taxa_evaporacao": random.uniform(*DOM_RHO)
        }
        r = executar_single(params)
        hist = r["historico"]
        if not hist.empty:
            hist["execucao"] = exec_id + 1
            historicos.append(hist)
            if "melhor_geracao" in hist.columns:
                idx_melhor = int(hist.melhor_geracao.idxmax())
                geracao_melhor = int(hist.loc[idx_melhor, "geracao"])
            else:
                geracao_melhor = None
        else:
            geracao_melhor = None
        resultados.append({
            "execucao": exec_id + 1,
            "melhor_qualidade_final": int(r["resultado"].get("valor_objetivo")) if r["resultado"].get("valor_objetivo") is not None else None,
            "geracao_melhor_global": geracao_melhor,
            "tempo_total_execucao": r["tempo"],
            "n_formigas": params["n_formigas"],
            "n_geracoes": params["n_geracoes"],
            "alfa": params["alfa"],
            "beta": params["beta"],
            "taxa_evaporacao": params["taxa_evaporacao"]
        })
    df_resumo = pd.DataFrame(resultados)
    df_hist = pd.concat(historicos, ignore_index=True) if historicos else pd.DataFrame()
    return df_resumo, df_hist


def estudo_lhs(n_amostras: int, n_geracoes: int, seed: int) -> tuple:
    """Executa estudo de sensibilidade via Latin Hypercube sampling."""
    inicializar_seed(seed)
    # Limites inferiores e superiores para parâmetros contínuos
    # n_formigas será tratado como inteiro (round / astype int)
    limites_inferiores = [10, 0, 0, 0.05]  # n_formigas, alfa, beta, taxa_evaporacao
    limites_superiores = [100, 7, 7, 0.5]
    sampler = qmc.LatinHypercube(d=len(limites_inferiores), seed=seed)
    sample = sampler.random(n_amostras)
    l_bounds = np.array(limites_inferiores)
    u_bounds = np.array(limites_superiores)
    amostras = qmc.scale(sample, l_bounds, u_bounds)
    cols = ["n_formigas", "alfa", "beta", "taxa_evaporacao"]
    df_params = pd.DataFrame(amostras, columns=cols)
    df_params["n_formigas"] = df_params["n_formigas"].round().astype(int)

    resultados = []
    historicos = []
    for exec_id, row in df_params.iterrows():
        params = {
            "n_formigas": int(row["n_formigas"]),
            "n_geracoes": n_geracoes,
            "alfa": float(row["alfa"]),
            "beta": float(row["beta"]),
            "taxa_evaporacao": float(row["taxa_evaporacao"])
        }
        r = executar_single(params)
        hist = r["historico"]
        if not hist.empty:
            hist["execucao"] = exec_id + 1
            historicos.append(hist)
            if "melhor_geracao" in hist.columns:
                idx_melhor = int(hist.melhor_geracao.idxmax())
                geracao_melhor = int(hist.loc[idx_melhor, "geracao"])
            else:
                geracao_melhor = None
        else:
            geracao_melhor = None
        resultados.append({
            "execucao": exec_id + 1,
            "melhor_qualidade_final": int(r["resultado"].get("valor_objetivo")) if r["resultado"].get("valor_objetivo") is not None else None,
            "geracao_melhor_global": geracao_melhor,
            "tempo_total_execucao": r["tempo"],
            "n_formigas": params["n_formigas"],
            "n_geracoes": params["n_geracoes"],
            "alfa": params["alfa"],
            "beta": params["beta"],
            "taxa_evaporacao": params["taxa_evaporacao"]
        })
    df_resumo = pd.DataFrame(resultados)
    df_hist = pd.concat(historicos, ignore_index=True) if historicos else pd.DataFrame()
    return df_resumo, df_hist


def salvar_resultados(df_resumo: pd.DataFrame, df_hist: pd.DataFrame, pasta_base: str, prefixo: str):
    os.makedirs(pasta_base, exist_ok=True)
    subdir = os.path.join(pasta_base, prefixo)
    os.makedirs(subdir, exist_ok=True)
    arq_resumo = os.path.join(subdir, f"{prefixo}_resumo.csv")
    arq_hist = os.path.join(subdir, f"{prefixo}_historico.csv")
    df_resumo.to_csv(arq_resumo, index=False)
    df_hist.to_csv(arq_hist, index=False)
    print("Arquivos gerados:")
    print(f" - {arq_resumo}")
    print(f" - {arq_hist}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gerar amostras de execução para ACOFast (convergência ou LHS)")
    parser.add_argument("--modo", choices=["convergencia", "lhs"], required=True, help="Tipo de estudo a executar")
    parser.add_argument("--n-execs", type=int, default=10, help="Número de execuções (modo convergencia)")
    parser.add_argument("--n-amostras", type=int, default=100, help="Número de amostras (modo lhs)")
    parser.add_argument("--geracoes", type=int, default=150, help="Número de gerações por execução")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    parser.add_argument("--saida", type=str, default="resultados_analises_simulacoes", help="Diretório base de saída")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.modo == "convergencia":
        df_resumo, df_hist = estudo_convergencia(args.n_execs, args.geracoes, args.seed)
        salvar_resultados(df_resumo, df_hist, args.saida, "aco_convergencia")
    else:
        df_resumo, df_hist = estudo_lhs(args.n_amostras, args.geracoes, args.seed)
        salvar_resultados(df_resumo, df_hist, args.saida, "aco_lhs")


if __name__ == "__main__":
    main()