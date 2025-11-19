def validar_config(alg_id: str, config: dict):
    """Valida parametros de config para cada algoritmo.
    Retorna (ok: bool, erros: list[str], avisos: list[str])."""
    erros = []
    avisos = []

    # Seed
    seed = config.get('SEED')
    if seed is not None:
        try:
            int(seed)
        except Exception:
            erros.append('Seed deve ser inteiro.')

    if alg_id == 'pli':
        w = config.get('PENALIDADE_W')
        if w is None:
            erros.append('PENALIDADE_W ausente.')
        else:
            try:
                w = float(w)
                if w <= 0:
                    erros.append('PENALIDADE_W deve ser > 0.')
                elif w > 1000:
                    avisos.append('PENALIDADE_W muito alto (>1000) pode distorcer objetivo.')
            except Exception:
                erros.append('PENALIDADE_W inválido (não numérico).')

    elif alg_id == 'aco':
        params = config.get('ACO_PARAMS', {})
        nf = params.get('n_formigas')
        ng = params.get('n_geracoes')
        alfa = params.get('alfa')
        beta = params.get('beta')
        evap = params.get('taxa_evaporacao')
        try:
            nf = int(nf)
            if nf <= 0: erros.append('n_formigas deve ser > 0.')
            elif nf > 2000: erros.append('n_formigas > 2000 não permitido.')
            elif nf > 500: avisos.append('n_formigas muito alto (>500) pode ser lento.')
        except Exception:
            erros.append('n_formigas inválido.')
        try:
            ng = int(ng)
            if ng <= 0: erros.append('n_geracoes deve ser > 0.')
            elif ng > 5000: erros.append('n_geracoes > 5000 não permitido.')
            elif ng > 1000: avisos.append('n_geracoes muito alto (>1000) pode ser lento.')
        except Exception:
            erros.append('n_geracoes inválido.')
        for nome,val in [('alfa',alfa),('beta',beta)]:
            try:
                v = float(val)
                if v < 0: erros.append(f'{nome} deve ser >= 0.')
                elif v > 20: avisos.append(f'{nome} alto (>20) pode gerar instabilidade.')
            except Exception:
                erros.append(f'{nome} inválido.')
        try:
            evap = float(evap)
            if not (0 < evap <= 1): erros.append('taxa_evaporacao deve estar em (0,1].')
            elif evap < 0.01: avisos.append('taxa_evaporacao muito baixa (<0.01) pode acumular feromônio excessivo.')
            elif evap > 0.9: avisos.append('taxa_evaporacao muito alta (>0.9) pode impedir convergência.')
        except Exception:
            erros.append('taxa_evaporacao inválida.')

    elif alg_id == 'ag':
        params = config.get('AG_PARAMS', {})
        npop = params.get('n_populacao')
        ng = params.get('n_geracoes')
        tx_cross = params.get('taxa_crossover')
        tx_mut = params.get('taxa_mutacao')
        tamanho_torneio = params.get('tamanho_torneio')
        fator_penalidade = params.get('fator_penalidade')
        try:
            npop = int(npop)
            if npop <= 0: erros.append('n_populacao deve ser > 0.')
            elif npop > 10000: erros.append('n_populacao > 10000 não permitido.')
            elif npop > 1000: avisos.append('n_populacao muito alto (>1000) pode ser lento.')
        except Exception:
            erros.append('n_populacao inválido.')
        try:
            ng = int(ng)
            if ng <= 0: erros.append('n_geracoes deve ser > 0.')
            elif ng > 10000: erros.append('n_geracoes > 10000 não permitido.')
            elif ng > 2000: avisos.append('n_geracoes muito alto (>2000) pode ser lento.')
        except Exception:
            erros.append('n_geracoes inválido.')
        try:
            tc = float(tx_cross)
            if not (0 < tc <= 1): erros.append('taxa_crossover deve estar em (0,1].')
            elif tc < 0.4: avisos.append('taxa_crossover baixa (<0.4) pode reduzir diversidade de recombinação.')
        except Exception:
            erros.append('taxa_crossover inválida.')
        try:
            tm = float(tx_mut)
            if not (0 < tm <= 1): erros.append('taxa_mutacao deve estar em (0,1].')
            elif tm > 0.4: avisos.append('taxa_mutacao alta (>0.4) pode gerar deriva aleatória.')
            elif tm < 0.001: avisos.append('taxa_mutacao muito baixa (<0.001) pode prejudicar exploração.')
        except Exception:
            erros.append('taxa_mutacao inválida.')
        try:
            tt = int(tamanho_torneio)
            if tt < 2: erros.append('tamanho_torneio deve ser >= 2.')
            elif npop and tt > npop: erros.append('tamanho_torneio não pode exceder n_populacao.')
        except Exception:
            erros.append('tamanho_torneio inválido.')
        try:
            fp = float(fator_penalidade)
            if fp < 0: erros.append('fator_penalidade deve ser >= 0.')
            elif fp > 1000: avisos.append('fator_penalidade muito alto (>1000) pode dominar fitness.')
        except Exception:
            erros.append('fator_penalidade inválido.')

    else:
        erros.append(f'Algoritmo desconhecido para validação: {alg_id}')

    return (len(erros) == 0, erros, avisos)

def validar_batch_items(items: list):
    """Valida lista de itens de batch. Para cada item (alg_id, config)."""
    erros_globais = []
    avisos_globais = []
    for idx, it in enumerate(items):
        ok, errs, avis = validar_config(it.get('alg_id'), it.get('config', {}))
        if not ok:
            erros_globais.append(f'Item {idx}: ' + '; '.join(errs))
        avisos_globais.extend(avis)
    return (len(erros_globais) == 0, erros_globais, avisos_globais)
