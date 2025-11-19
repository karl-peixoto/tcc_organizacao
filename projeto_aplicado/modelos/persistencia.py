import json
import time
from pathlib import Path
import csv
from typing import Dict, Any, List

# Diretório base para armazenar histórico e alocações detalhadas
RESULTADOS_DIR = Path(__file__).parent.parent.parent / "resultados"
HISTORICO_CSV = RESULTADOS_DIR / "historico_execucoes.csv"

_CAMPOS_PADRAO = [
    "id_execucao",
    "timestamp",
    "algoritmo",
    "valor_objetivo",
    "soma_preferencias",
    "penalidade_total",
    "num_alocacoes_preferencia_zero",
    "tempo_execucao",
    "seed",
    "config_json",
    "metricas_iteracao_json",
    "alocacao_csv"
]

def inicializar():
    """Garante criação do diretório e arquivo CSV de histórico."""
    RESULTADOS_DIR.mkdir(exist_ok=True)
    if not HISTORICO_CSV.exists():
        with HISTORICO_CSV.open(mode="w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(_CAMPOS_PADRAO)

def _gerar_id_execucao() -> str:
    """Gera identificador único baseado em timestamp de alta resolução."""
    return str(int(time.time() * 1000))

def salvar_resultado(resultado: Dict[str, Any], parametros: Dict[str, Any]) -> str:
    """Salva resultado de execução em CSV principal e grava alocação detalhada em arquivo separado.

    Args:
        resultado: dicionário padronizado contendo chaves como 'nome_algoritmo', 'valor_objetivo', 'alocacao', etc.
        parametros: dicionário de parâmetros usados (config do algoritmo, hiperparâmetros, seed, etc.).

    Returns:
        id_execucao gerado para referencia futura.
    """
    inicializar()

    id_execucao = _gerar_id_execucao()
    timestamp = int(time.time())

    algoritmo = resultado.get('nome_algoritmo') or parametros.get('algoritmo') or 'DESCONHECIDO'
    valor_objetivo = resultado.get('valor_objetivo')
    soma_preferencias = resultado.get('soma_preferencias')
    penalidade_total = resultado.get('penalidade_total')
    num_zero = resultado.get('num_alocacoes_preferencia_zero')
    tempo_execucao = resultado.get('tempo')
    seed = resultado.get('seed') or parametros.get('config', {}).get('SEED')
    config_json = json.dumps(parametros, ensure_ascii=False)
    metricas_iteracao_json = json.dumps(resultado.get('metricas_iteracao', []), ensure_ascii=False)

    # Salva alocação detalhada em arquivo próprio
    alocacao_registros = resultado.get('alocacao', [])
    nome_alocacao_csv = f"alocacao_{id_execucao}.csv"
    caminho_alocacao = RESULTADOS_DIR / nome_alocacao_csv
    if alocacao_registros:
        # Descobre colunas dinamicamente
        colunas = sorted({chave for linha in alocacao_registros for chave in linha.keys()})
        with caminho_alocacao.open(mode='w', newline='', encoding='utf-8') as f_aloc:
            writer = csv.DictWriter(f_aloc, fieldnames=colunas)
            writer.writeheader()
            writer.writerows(alocacao_registros)
    else:
        # Cria arquivo vazio com cabeçalho padrão minimal
        with caminho_alocacao.open(mode='w', newline='', encoding='utf-8') as f_aloc:
            writer = csv.writer(f_aloc)
            writer.writerow(['id_disciplina','id_docente','preferencia'])

    # Append no histórico
    with HISTORICO_CSV.open(mode="a", newline='', encoding='utf-8') as f_hist:
        writer = csv.writer(f_hist)
        writer.writerow([
            id_execucao,
            timestamp,
            algoritmo,
            valor_objetivo,
            soma_preferencias,
            penalidade_total,
            num_zero,
            tempo_execucao,
            seed,
            config_json,
            metricas_iteracao_json,
            nome_alocacao_csv
        ])

    return id_execucao

def listar_historico(limit: int = None) -> List[Dict[str, Any]]:
    """Lê histórico completo (ou limitado) e retorna lista de dicionários."""
    if not HISTORICO_CSV.exists():
        return []
    resultados = []
    with HISTORICO_CSV.open(mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            resultados.append(row)
            if limit is not None and i + 1 >= limit:
                break
    return resultados

def carregar_alocacao(id_execucao: str) -> List[Dict[str, Any]]:
    """Carrega o arquivo de alocação associado a um id_execucao."""
    caminho = RESULTADOS_DIR / f"alocacao_{id_execucao}.csv"
    if not caminho.exists():
        return []
    with caminho.open(mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)
