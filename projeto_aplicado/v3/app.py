from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from io import BytesIO
from datetime import datetime
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJETO_APLICADO_DIR = THIS_FILE.parent.parent  # .../projeto_aplicado
ROOT_DIR = PROJETO_APLICADO_DIR.parent          # raiz do repo

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Importa otimizadores existentes
from projeto_aplicado.modelos.otimizador_pli import OtimizadorPLI
from projeto_aplicado.modelos.otimizador_aco import OtimizadorACO
from projeto_aplicado.modelos.otimizador_ag import OtimizadorAG
from projeto_aplicado.modelos.analisador import AnalisadorDeSolucao

app = Flask(__name__)

# Configuração mínima para arquivos (reusa estrutura v2)
ARQUIVOS_DADOS = {
    "professores": "docentes.csv",
    "disciplinas": "disciplinas.csv",
    "preferencias": "preferencias.csv",
    "conflitos": "matriz_conflitos.csv"
}

# Estado simples em memória
_cache_dados = None
_ultimo_resultado = None
_ultima_analise = None


def carregar_dados():
    global _cache_dados
    if _cache_dados is not None:
        return _cache_dados
    base = "dados"
    df_prof = pd.read_csv(f"{base}/docentes.csv")
    df_disc = pd.read_csv(f"{base}/disciplinas.csv")
    df_pref = pd.read_csv(f"{base}/preferencias.csv")
    df_conf = pd.read_csv(f"{base}/matriz_conflitos.csv")

    # Padronizações simples
    for col in ["id_docente", "nome_docente"]:
        if col in df_prof.columns:
            df_prof[col] = df_prof[col].astype(str).str.strip()
    for col in ["id_disciplina", "nome_disciplina"]:
        if col in df_disc.columns:
            df_disc[col] = df_disc[col].astype(str).str.strip()
    df_pref['id_docente'] = df_pref['id_docente'].astype(str).str.strip()
    df_pref['id_disciplina'] = df_pref['id_disciplina'].astype(str).str.strip()

    # Merge para visão tabular inicial (preferencias respondidas)
    df_merge = df_pref.merge(df_prof, on='id_docente', how='left')\
                      .merge(df_disc, on='id_disciplina', how='left')

    # Renomear / criar colunas esperadas
    # ch_max vem de carga_maxima
    if 'carga_maxima' in df_prof.columns:
        df_merge['ch_max'] = df_merge['carga_maxima']
    # ch_disciplina de carga_horaria
    if 'carga_horaria' in df_disc.columns:
        df_merge['ch_disciplina'] = df_merge['carga_horaria']
    # codigo_turma: se existir na disciplinas, senão usa id_disciplina
    if 'codigo_turma' not in df_merge.columns:
        df_merge['codigo_turma'] = df_merge['id_disciplina']
    # horario_extenso: se não existir, cria placeholder
    if 'horario_extenso' not in df_merge.columns:
        df_merge['horario_extenso'] = 'N/D'

    _cache_dados = {
        'professores': df_prof,
        'disciplinas': df_disc,
        'preferencias': df_pref,
        'conflitos': df_conf,
        'merge': df_merge
    }
    return _cache_dados


def resumo_preferencias(df_merge):
    # Contagem por preferencia
    contagem = df_merge['preferencia'].value_counts().to_dict()
    return {
        'pref_3': contagem.get(3, 0),
        'pref_2': contagem.get(2, 0),
        'pref_1': contagem.get(1, 0),
        'pref_0': contagem.get(0, 0)
    }


@app.route('/')
def dados_iniciais():
    dados = carregar_dados()
    df_prof = dados['professores']
    df_disc = dados['disciplinas']
    df_pref = dados['preferencias']
    df_merge = dados['merge']

    # Lista de professores com nome capitalizado, ch_max e contagens de preferências
    pref_counts = df_pref.groupby(['id_docente', 'preferencia']).size().unstack(fill_value=0)
    # Garantir colunas 0..3
    for nivel in [0, 1, 2, 3]:
        if nivel not in pref_counts.columns:
            pref_counts[nivel] = 0
    professores = []
    for _, row in df_prof.iterrows():
        id_doc = row.get('id_docente')
        # Detecta coluna de nome disponível (sem alterar capitalização original)
        nome_bruto = (
            row.get('nome_docente') if 'nome_docente' in df_prof.columns else
            row.get('docente') if 'docente' in df_prof.columns else
            row.get('nome') if 'nome' in df_prof.columns else
            row.get('Nome') if 'Nome' in df_prof.columns else ''
        )
        nome = str(nome_bruto).strip()
        ch_max = row.get('carga_maxima') or row.get('ch_max')
        professores.append({
            'id_docente': id_doc,
            'nome': nome,
            'ch_max': ch_max,
            'pref_0': int(pref_counts.at[id_doc, 0]) if id_doc in pref_counts.index else 0,
            'pref_1': int(pref_counts.at[id_doc, 1]) if id_doc in pref_counts.index else 0,
            'pref_2': int(pref_counts.at[id_doc, 2]) if id_doc in pref_counts.index else 0,
            'pref_3': int(pref_counts.at[id_doc, 3]) if id_doc in pref_counts.index else 0,
        })

    # Lista de disciplinas com campos originais + horario, horario_extenso, codigo_turma
    if 'horario' not in df_disc.columns:
        df_disc['horario'] = 'N/D'
    if 'horario_extenso' not in df_disc.columns:
        df_disc['horario_extenso'] = 'N/D'
    if 'codigo_turma' not in df_disc.columns:
        df_disc['codigo_turma'] = df_disc['id_disciplina']
    disciplinas = df_disc.to_dict(orient='records')

    resumo = resumo_preferencias(df_merge)
    return render_template(
        'dados_iniciais.html',
        tabela=df_merge.head(200).to_dict(orient='records'),
        resumo=resumo,
        professores=professores,
        disciplinas=disciplinas
    )

@app.route('/dados')
def dados_alias():
    # Alias para mesma página, conforme requisito de rota /dados
    return dados_iniciais()


@app.route('/executar', methods=['GET', 'POST'])
def executar():
    global _ultimo_resultado, _ultima_analise
    if request.method == 'GET':
        dados = carregar_dados()
        professores = dados['professores'].to_dict(orient='records')
        disciplinas = dados['disciplinas'].to_dict(orient='records')
        defaults = {
            'pli': {'PENALIDADE_W': 4},
            'aco': {'ACO_PARAMS': {'n_formigas': 8, 'n_geracoes': 30, 'alfa': 1.0, 'beta': 2.0, 'taxa_evaporacao': 0.15}},
            'ag': {'AG_PARAMS': {'n_populacao': 40, 'n_geracoes': 60, 'taxa_crossover': 0.8, 'taxa_mutacao': 0.03, 'tamanho_torneio': 3, 'fator_penalidade': 800}}
        }
        return render_template('executar.html', defaults=defaults, professores=professores, disciplinas=disciplinas)

    algoritmos_selecionados = request.form.getlist('algoritmos')
    if not algoritmos_selecionados:
        return "Nenhum algoritmo selecionado", 400

    # Alocações fixas (JSON serializado no textarea oculto)
    import json
    alocacoes_str = request.form.get('alocacoes_fixas', '[]')
    try:
        alocacoes_raw = json.loads(alocacoes_str)
        if not isinstance(alocacoes_raw, list):
            alocacoes_raw = []
    except Exception:
        alocacoes_raw = []
    # Normaliza para formato esperado (professor, disciplina)
    alocacoes_fixas = []
    for item in alocacoes_raw:
        prof = item.get('professor') or item.get('id_docente')
        disc = item.get('disciplina') or item.get('id_disciplina')
        if prof and disc:
            alocacoes_fixas.append({'professor': str(prof).strip(), 'disciplina': str(disc).strip()})

    seed = request.form.get('seed')
    seed_val = None
    if seed:
        try:
            seed_val = int(seed)
        except Exception:
            seed_val = None

    ultimo_df = None
    for alg in algoritmos_selecionados:
        config_base = {
            'ARQUIVOS_DADOS': ARQUIVOS_DADOS,
            'SEED': seed_val,
            'ALOCACOES_FIXAS': alocacoes_fixas,
            'PREFERENCIA_SERVICO': 3,
            'PREFERENCIA_DEMAIS': 0
        }
        if alg == 'pli':
            w = request.form.get('pli_w')
            try:
                config_base['PENALIDADE_W'] = float(w) if w is not None else 4.0
            except Exception:
                config_base['PENALIDADE_W'] = 4.0
            otimizador = OtimizadorPLI(config_base)
        elif alg == 'aco':
            def _parse_int(name, default):
                try:
                    return int(request.form.get(name, default))
                except Exception:
                    return default
            def _parse_float(name, default):
                try:
                    return float(request.form.get(name, default))
                except Exception:
                    return default
            config_base['ACO_PARAMS'] = {
                'n_formigas': _parse_int('aco_n_formigas', 8),
                'n_geracoes': _parse_int('aco_n_geracoes', 30),
                'alfa': _parse_float('aco_alfa', 1.0),
                'beta': _parse_float('aco_beta', 2.0),
                'taxa_evaporacao': _parse_float('aco_taxa_evaporacao', 0.15)
            }
            otimizador = OtimizadorACO(config_base)
        elif alg == 'ag':
            def _pi(name, default):
                try:
                    return int(request.form.get(name, default))
                except Exception:
                    return default
            def _pf(name, default):
                try:
                    return float(request.form.get(name, default))
                except Exception:
                    return default
            config_base['AG_PARAMS'] = {
                'n_populacao': _pi('ag_n_populacao', 40),
                'n_geracoes': _pi('ag_n_geracoes', 60),
                'taxa_crossover': _pf('ag_taxa_crossover', 0.8),
                'taxa_mutacao': _pf('ag_taxa_mutacao', 0.03),
                'tamanho_torneio': _pi('ag_tamanho_torneio', 3),
                'fator_penalidade': _pi('ag_fator_penalidade', 800)
            }
            otimizador = OtimizadorAG(config_base)
        else:
            continue

        resultado = otimizador.resolver()
        if not resultado or resultado.get('alocacao_final') is None:
            continue
        df_aloc = resultado['alocacao_final'].copy()
        dados = carregar_dados()
        df_prof = dados['professores']
        df_disc = dados['disciplinas']
        # Ajusta nomes das colunas conforme dataset real
        col_nome_prof = 'nome_docente' if 'nome_docente' in df_prof.columns else ('docente' if 'docente' in df_prof.columns else None)
        col_nome_disc = 'nome_disciplina' if 'nome_disciplina' in df_disc.columns else ('disciplina' if 'disciplina' in df_disc.columns else None)
        cols_prof = ['id_docente', 'carga_maxima'] + ([col_nome_prof] if col_nome_prof else [])
        cols_disc = ['id_disciplina', 'carga_horaria'] + ([col_nome_disc] if col_nome_disc else [])
        df_aloc = df_aloc.merge(df_prof[cols_prof], on='id_docente', how='left')\
                         .merge(df_disc[cols_disc + ([c for c in ['horario'] if c in df_disc.columns])], on='id_disciplina', how='left')
        df_aloc['ch_max'] = df_aloc['carga_maxima']
        df_aloc['ch_disciplina'] = df_aloc['carga_horaria']
        if 'codigo_turma' not in df_aloc.columns:
            df_aloc['codigo_turma'] = df_aloc['id_disciplina']
        if 'horario_extenso' not in df_aloc.columns:
            df_aloc['horario_extenso'] = 'N/D'
        # Marca alocações fixas (roxo no resultado)
        fix_set = {(a['professor'], a['disciplina']) for a in alocacoes_fixas}
        df_aloc['fixada'] = df_aloc.apply(lambda r: (str(r['id_docente']), str(r['id_disciplina'])) in fix_set, axis=1)
        ultimo_df = df_aloc

    if ultimo_df is None:
        return "Falha na otimização", 500

    _ultimo_resultado = ultimo_df
    analisador = AnalisadorDeSolucao({'ARQUIVOS_DADOS': ARQUIVOS_DADOS})
    try:
        _ultima_analise = analisador.avaliar(ultimo_df[['id_docente', 'id_disciplina']])
    except Exception:
        _ultima_analise = None
    return redirect(url_for('resultado'))


@app.route('/resultado')
def resultado():
    if _ultimo_resultado is None:
        return redirect(url_for('executar'))

    df_view = _ultimo_resultado.copy()

    # Ordena por preferencia (desc) e professor
    if 'preferencia' in df_view.columns:
        df_view.sort_values(['preferencia', 'id_docente'], ascending=[False, True], inplace=True)

    registros = df_view.to_dict(orient='records')
    analise = _ultima_analise or {}

    # Resumo das preferências e fixadas
    resumo = {
        'escore_total': analise.get('escore_total'),
        'pref_3': 0,
        'pref_2': 0,
        'pref_1': 0,
        'pref_0': 0,
        'fixadas': 0,
        'total': len(df_view)
    }
    if 'preferencia' in df_view.columns:
        vc = df_view['preferencia'].value_counts().to_dict()
        for k in [0,1,2,3]:
            resumo[f'pref_{k}'] = vc.get(k, 0)
    if 'fixada' in df_view.columns:
        try:
            resumo['fixadas'] = int(df_view['fixada'].sum())
        except Exception:
            resumo['fixadas'] = 0

    # --- Construção dos mapas para interatividade (hover) ---
    # Mapa professor -> lista de disciplinas alocadas
    dados = carregar_dados()
    df_disc = dados['disciplinas']
    # Detectar coluna de nome de disciplina
    col_nome_disc = 'nome_disciplina' if 'nome_disciplina' in df_disc.columns else ('disciplina' if 'disciplina' in df_disc.columns else None)
    # Cria lista de nomes de disciplina alocadas por professor
    if col_nome_disc:
        mapa_prof = df_view.groupby('id_docente')[col_nome_disc].apply(list).to_dict()
    else:
        mapa_prof = df_view.groupby('id_docente')['id_disciplina'].apply(list).to_dict()

    # DataFrames auxiliares
    df_pref = dados['preferencias']
    df_prof = dados['professores']
    col_nome_prof = 'nome_docente' if 'nome_docente' in df_prof.columns else ('docente' if 'docente' in df_prof.columns else None)

    # --- Mapa disciplina -> professores com preferência 3 ---
    df_pref3 = df_pref[df_pref['preferencia'] == 3].copy()
    if col_nome_prof:
        df_pref3_prof = df_pref3.merge(df_prof[['id_docente', col_nome_prof]], on='id_docente', how='left')
        mapa_disc = df_pref3_prof.groupby('id_disciplina')[col_nome_prof].apply(lambda x: [str(n).strip() for n in x if pd.notnull(n)]).to_dict()
    else:
        mapa_disc = df_pref3.groupby('id_disciplina')['id_docente'].apply(list).to_dict()
    for d_id in df_disc['id_disciplina'].tolist():  # garante chaves vazias
        mapa_disc.setdefault(d_id, [])

    # --- Mapa professor -> disciplinas com preferência 3 (deduplicadas por nome) ---
    if col_nome_disc:
        df_pref3_disc = df_pref3.merge(df_disc[['id_disciplina', col_nome_disc]], on='id_disciplina', how='left')
        # Remove duplicatas de nomes preservando ordem (caso haja múltiplos códigos para o mesmo nome)
        mapa_prof_pref3 = df_pref3_disc.groupby('id_docente')[col_nome_disc].apply(
            lambda s: list(dict.fromkeys([str(n).strip() for n in s if pd.notnull(n)]))
        ).to_dict()
    else:
        # Sem coluna de nome, usa id_disciplina e dedup também
        mapa_prof_pref3 = df_pref3.groupby('id_docente')['id_disciplina'].apply(
            lambda s: list(dict.fromkeys([str(n).strip() for n in s if pd.notnull(n)]))
        ).to_dict()
    for pid in df_prof['id_docente'].tolist():  # garante chaves vazias
        mapa_prof_pref3.setdefault(pid, [])

    return render_template(
        'resultado.html',
        registros=registros,
        analise=analise,
        resumo=resumo,
        prof_alloc_map=mapa_prof,
        disc_pref3_map=mapa_disc,
        prof_pref3_map=mapa_prof_pref3
    )


@app.route('/resultado/export')
def resultado_export():
    if _ultimo_resultado is None:
        return "Nada para exportar", 400
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        _ultimo_resultado.to_excel(writer, index=False, sheet_name='alocacao')
    output.seek(0)
    nome = f"alocacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(output, as_attachment=True, download_name=nome, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route('/health')
def health():
    return {"status": "ok", "versao": "v3"}


if __name__ == '__main__':
    app.run(debug=True)
