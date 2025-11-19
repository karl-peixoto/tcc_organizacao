import threading
import time
import uuid
import pandas as pd
from typing import Dict, Any, Callable
from projeto_aplicado.modelos import otimizador_aco, otimizador_ag, otimizador_pli, persistencia

ESTADOS: Dict[str, Dict[str, Any]] = {}
GRUPOS: Dict[str, Dict[str, Any]] = {}
_analisador = None

def configurar(analisador):
    global _analisador
    _analisador = analisador

def _resolver_factory(alg_id: str, config: dict):
    alg_id = alg_id.lower()
    if alg_id == 'aco':
        return 'ACO', otimizador_aco.OtimizadorACO(config)
    if alg_id == 'ag':
        return 'AG', otimizador_ag.OtimizadorAG(config)
    if alg_id == 'pli':
        return 'PLI', otimizador_pli.OtimizadorPLI(config)
    raise ValueError(f"Algoritmo desconhecido: {alg_id}")

def iniciar_job(alg_id: str, config: dict) -> str:
    job_id = str(uuid.uuid4())
    total_iter = _inferir_total_iteracoes(alg_id, config)
    ESTADOS[job_id] = {
        'status': 'running',
        'algoritmo': alg_id.upper(),
        'progress': 0.0,
        'iteracao_atual': 0,
        'total_iteracoes': total_iter,
        'melhor_objetivo': None,
        'metricas_ultimas': [],  # últimas 30 para resposta rápida
        'id_execucao_final': None,
        'erro': None
    }
    t = threading.Thread(target=_thread_job, args=(job_id, alg_id, config), daemon=True)
    t.start()
    return job_id

def iniciar_batch(definicoes: Dict[str, Any]) -> str:
    """Inicia um lote de execuções.
    definicoes: {
        'items': [ {'alg_id': 'aco', 'config': {...}}, ... ]
        'meta': {optional info}
    }
    Retorna group_id.
    """
    group_id = str(uuid.uuid4())
    itens_def = definicoes.get('items', [])
    job_ids = []
    itens_map = {}
    for item in itens_def:
        alg = item['alg_id']
        cfg = item['config']
        # Injeta group_id no config (para rastreio em persistência via config_json)
        cfg['GROUP_ID'] = group_id
        jid = iniciar_job(alg, cfg)
        job_ids.append(jid)
        itens_map[jid] = {
            'algoritmo': alg.upper(),
            'seed': cfg.get('SEED'),
            'status': 'running',
            'melhor_objetivo': None,
            'progress': 0.0
        }
    GRUPOS[group_id] = {
        'job_ids': job_ids,
        'items': itens_map,
        'status': 'running',
        'meta': definicoes.get('meta', {}),
        'created_at': time.time()
    }
    return group_id

def _inferir_total_iteracoes(alg_id: str, config: dict) -> int:
    alg_id = alg_id.lower()
    if alg_id == 'aco':
        return config.get('ACO_PARAMS', {}).get('n_geracoes', 1)
    if alg_id == 'ag':
        return config.get('AG_PARAMS', {}).get('n_geracoes', 1)
    return 1  # PLI

def _callback_iteracao(job_id: str, metrica: Dict[str, Any]):
    estado = ESTADOS.get(job_id)
    if not estado or estado['status'] != 'running':
        return
    iteracao = metrica.get('geracao') or metrica.get('iteracao') or (estado['iteracao_atual'] + 1)
    estado['iteracao_atual'] = iteracao
    total = estado['total_iteracoes'] or 1
    estado['progress'] = min(1.0, iteracao / total)
    melhor_global = metrica.get('melhor_global') or metrica.get('melhor_geracao')
    if melhor_global is not None:
        estado['melhor_objetivo'] = melhor_global
    ult = estado['metricas_ultimas']
    ult.append(metrica)
    if len(ult) > 30:
        ult.pop(0)

def _thread_job(job_id: str, alg_id: str, config: dict):
    inicio = time.time()
    try:
        nome_alg, instancia = _resolver_factory(alg_id, config)
        resultado = instancia.resolver(callback_iteracao=lambda m: _callback_iteracao(job_id, m))
        # Adiciona análise
        if _analisador and resultado.get('alocacao_final') is not None:
            try:
                df = resultado['alocacao_final']
                resultado['analise'] = _analisador.avaliar(df)
            except Exception as e:
                print(f"Falha ao analisar solução assíncrona: {e}")
        # Serializa alocação final para persistência
        resultado_simplificado = {
            'nome_algoritmo': nome_alg,
            'tempo': time.time() - inicio,
            'valor_objetivo': resultado.get('valor_objetivo'),
            'alocacao': resultado.get('alocacao_final', pd.DataFrame()).to_dict(orient='records'),
            'soma_preferencias': resultado.get('soma_preferencias'),
            'num_alocacoes_preferencia_zero': resultado.get('num_alocacoes_preferencia_zero'),
            'penalidade_total': resultado.get('penalidade_total'),
            'metricas_iteracao': resultado.get('metricas_iteracao', []),
            'seed': resultado.get('seed'),
            'analise': resultado.get('analise')
        }
        id_execucao = persistencia.salvar_resultado(resultado_simplificado, {'algoritmo': nome_alg, 'config': config})
        estado = ESTADOS[job_id]
        estado['status'] = 'done'
        estado['progress'] = 1.0
        estado['id_execucao_final'] = id_execucao
        if resultado_simplificado.get('valor_objetivo') is not None:
            estado['melhor_objetivo'] = resultado_simplificado['valor_objetivo']
    except Exception as e:
        estado = ESTADOS.get(job_id)
        if estado:
            estado['status'] = 'error'
            estado['erro'] = str(e)
        print(f"Erro no job {job_id}: {e}")

def obter_estado(job_id: str) -> Dict[str, Any]:
    return ESTADOS.get(job_id)

def obter_estado_grupo(group_id: str) -> Dict[str, Any]:
    grupo = GRUPOS.get(group_id)
    if not grupo:
        return None
    # Atualiza itens com estado vivo
    for jid in grupo['job_ids']:
        estado_job = ESTADOS.get(jid)
        if not estado_job:
            continue
        item = grupo['items'].get(jid, {})
        item['status'] = estado_job['status']
        item['progress'] = estado_job.get('progress', 0.0)
        item['melhor_objetivo'] = estado_job.get('melhor_objetivo')
        grupo['items'][jid] = item
    job_statuses = [ESTADOS[j] for j in grupo['job_ids'] if j in ESTADOS]
    total = len(job_statuses)
    concluidos = sum(1 for s in job_statuses if s['status'] == 'done')
    erros = [s for s in job_statuses if s['status'] == 'error']
    if total > 0 and (concluidos + len(erros) == total):
        grupo['status'] = 'done'
    progresso_medio = 0.0
    if total > 0:
        progresso_medio = sum(s.get('progress', 0) for s in job_statuses) / total
    melhor_objetivos = [s.get('melhor_objetivo') for s in job_statuses if s.get('melhor_objetivo') is not None]
    resumo = {
        'group_id': group_id,
        'status': grupo['status'],
        'progress': round(progresso_medio * 100, 2),  # percentual para UI
        'progresso_medio': progresso_medio,
        'total_jobs': total,
        'concluidos': concluidos,
        'erros': len(erros),
        'melhores_objetivos': melhor_objetivos,
        'items': grupo['items']
    }
    return resumo
