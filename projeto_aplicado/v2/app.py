# app.py (na raiz do projeto)

import pandas as pd
import os
import time
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from pathlib import Path
import json  # Para lidar com alocações fixas do formulário
import sys

# --- Ajuste de PATH para garantir import de config (raiz do repo) e modelos ---
THIS_FILE = Path(__file__).resolve()
V2_DIR = THIS_FILE.parent
PROJETO_APLICADO_DIR = V2_DIR.parent
REPO_ROOT = PROJETO_APLICADO_DIR.parent

# Garante raiz no sys.path antes de importar config
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # Importa o config.py da raiz

# Adiciona pasta modelos ao path explícito (opcional pois pacote, mas garante)
MODELOS_DIR = PROJETO_APLICADO_DIR / 'modelos'
if str(MODELOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELOS_DIR))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from projeto_aplicado.modelos import otimizador_pli, otimizador_aco, otimizador_ag, otimizador_base, persistencia
from projeto_aplicado.modelos.analisador import AnalisadorDeSolucao
from projeto_aplicado.v2 import tarefas, validacao


app = Flask(__name__)
# Chave secreta para usar 'flash' e 'session' (necessário para passar resultados entre rotas)
app.secret_key = 'tcc_211028972'
analisador_global = AnalisadorDeSolucao(config={'ARQUIVOS_DADOS': config.CONFIG_BASE_ARQUIVOS})
tarefas.configurar(analisador_global)

# ---------------- Dashboard Dados (Nova API de listagem detalhada) ----------------
import unicodedata
from flask import jsonify

_DASHBOARD_CACHE = {
    'df': None,
    'last_load': 0
}

def _normalize_txt(s: str) -> str:
    if s is None:
        return ''
    s = str(s).lower().strip()
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])

def _load_dashboard_df(force=False):
    """Carrega e cacheia DataFrame detalhado com professor-disciplina-preferencia.
    Columns finais:
      id_docente, docente, id_disciplina, disciplina, preferencia, ch_disciplina?, horario?, busca_texto
    """
    import time as _t
    if _DASHBOARD_CACHE['df'] is not None and not force:
        return _DASHBOARD_CACHE['df']
    try:
        df_prof = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['professores'])
        df_pref = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['preferencias'])
        df_disc = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['disciplinas'])
    except FileNotFoundError as e:
        raise RuntimeError(f"Arquivo não encontrado: {e}")
    except Exception as e:
        raise RuntimeError(f"Erro ao ler CSV: {e}")

    # Limpeza básica
    for df in [df_prof, df_pref, df_disc]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()

    # Merge preferências + professores
    df = df_pref.merge(df_prof, on='id_docente', how='left', suffixes=('','_prof'))
    # Merge disciplinas
    df = df.merge(df_disc, on='id_disciplina', how='left', suffixes=('','_disc'))

    # Campos esperados opcionais
    if 'ch_disciplina' not in df.columns:
        df['ch_disciplina'] = None
    # Horário (se não existir, cria vazio)
    horario_col = None
    for possivel in ['horario','hora','periodo']:
        if possivel in df.columns:
            horario_col = possivel
            break
    if not horario_col:
        df['horario'] = None
        horario_col = 'horario'

    # Normalização texto de busca
    df['busca_texto'] = (
        df.get('docente','').fillna('') + ' | ' +
        df.get('disciplina','').fillna('') + ' | ' +
        df.get(horario_col,'').fillna('')
    ).apply(_normalize_txt)

    # Armazena em cache
    _DASHBOARD_CACHE['df'] = df
    _DASHBOARD_CACHE['last_load'] = _t.time()
    return df

_SORTABLE_FIELDS = {
    'docente': 'docente',
    'disciplina': 'disciplina',
    'preferencia': 'preferencia',
    'horario': 'horario',
    'ch_disciplina': 'ch_disciplina'
}

def _apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    filtro_q = args.get('q')
    filtro_prof = args.get('professor')
    filtro_disc = args.get('disciplina')
    filtro_hor = args.get('horario')

    result = df
    if filtro_prof:
        fp_norm = _normalize_txt(filtro_prof)
        mask_prof = result['docente'].apply(lambda x: fp_norm in _normalize_txt(x)) | (result['id_docente'].astype(str) == filtro_prof)
        result = result[mask_prof]
    if filtro_disc:
        fd_norm = _normalize_txt(filtro_disc)
        mask_disc = result['disciplina'].apply(lambda x: fd_norm in _normalize_txt(x)) | (result['id_disciplina'].astype(str) == filtro_disc)
        result = result[mask_disc]
    if filtro_hor:
        fh_norm = _normalize_txt(filtro_hor)
        if 'horario' in result.columns:
            mask_hor = result['horario'].apply(lambda x: fh_norm in _normalize_txt(x))
            result = result[mask_hor]
    if filtro_q:
        fq_norm = _normalize_txt(filtro_q)
        mask_q = result['busca_texto'].str.contains(fq_norm, na=False)
        result = result[mask_q]
    return result

def _apply_order(df: pd.DataFrame, sort_key: str, order: str) -> pd.DataFrame:
    col = _SORTABLE_FIELDS.get(sort_key, 'docente')
    asc = (order != 'desc')
    try:
        return df.sort_values(by=col, ascending=asc, kind='stable')
    except Exception:
        return df

def _paginate(df: pd.DataFrame, page: int, page_size: int):
    total = len(df)
    if page < 1: page = 1
    if page_size < 1: page_size = 50
    if page_size > 500: page_size = 500
    start = (page - 1) * page_size
    end = start + page_size
    fatia = df.iloc[start:end]
    pages = (total // page_size) + (1 if total % page_size else 0)
    return total, pages, fatia

@app.get('/dashboard/dados')
def dashboard_dados():
    try:
        df = _load_dashboard_df()
    except RuntimeError as e:
        return jsonify({'erro': str(e)}), 500
    args = request.args
    filtrado = _apply_filters(df, args)
    sort_key = args.get('sort','docente')
    order = args.get('order','asc')
    ordenado = _apply_order(filtrado, sort_key, order)
    try:
        page = int(args.get('page', '1'))
    except ValueError:
        page = 1
    try:
        page_size = int(args.get('page_size','50'))
    except ValueError:
        page_size = 50
    total, pages, fatia = _paginate(ordenado, page, page_size)
    # Selecionar colunas para saída
    cols_out = ['id_docente','docente','id_disciplina','disciplina','preferencia','ch_disciplina']
    if 'horario' in fatia.columns:
        cols_out.append('horario')
    rows = fatia[cols_out].to_dict(orient='records')
    return jsonify({
        'total': total,
        'pages': pages,
        'page': page,
        'page_size': page_size,
        'sort': sort_key,
        'order': order,
        'rows': rows
    })

@app.get('/dashboard/dados/export')
def dashboard_dados_export():
    try:
        df = _load_dashboard_df()
    except RuntimeError as e:
        return jsonify({'erro': str(e)}), 500
    args = request.args
    filtrado = _apply_filters(df, args)
    sort_key = args.get('sort','docente')
    order = args.get('order','asc')
    ordenado = _apply_order(filtrado, sort_key, order)
    cols_out = ['id_docente','docente','id_disciplina','disciplina','preferencia','ch_disciplina']
    if 'horario' in ordenado.columns:
        cols_out.append('horario')
    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(cols_out)
    for _, row in ordenado[cols_out].iterrows():
        writer.writerow([row.get(c) for c in cols_out])
    resp = Response(output.getvalue(), mimetype='text/csv')
    resp.headers['Content-Disposition'] = 'attachment; filename=dashboard_dados.csv'
    return resp

# --- Função Auxiliar para Carregar Dados do Dashboard ---
def carregar_dados_dashboard():
    dados = {}
    try:
        # Carrega Professores
        df_docentes = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['professores'])
        df_preferencias = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['preferencias'])
        df_docentes['id_docente'] = df_docentes['id_docente'].str.strip()
        df_preferencias['id_docente'] = df_preferencias['id_docente'].str.strip()
        pref_counts = df_preferencias.groupby('id_docente')['preferencia'].value_counts().unstack(fill_value=0)
        pref_counts.columns = [f'pref_{int(col)}' for col in pref_counts.columns]
        df_docentes_completo = df_docentes.merge(pref_counts, on='id_docente', how='left').fillna(0)
        for col in ['pref_3', 'pref_2', 'pref_1', 'pref_0']:
             if col not in df_docentes_completo.columns: df_docentes_completo[col] = 0
        dados['professores'] = df_docentes_completo.to_dict(orient='records')

        # Carrega Disciplinas
        df_disciplinas = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['disciplinas'])
        dados['disciplinas'] = df_disciplinas.to_dict(orient='records')

    except FileNotFoundError as e:
        flash(f"Erro ao carregar dados: {e}. Verifique os arquivos na pasta 'dados'.", "danger")
    except Exception as e:
        flash(f"Erro inesperado ao processar dados: {e}", "danger")
    return dados

# --- Rotas da Aplicação ---

@app.route('/')
def dashboard():
    """Tela inicial: Visualização de Professores e Disciplinas."""
    dados_dash = carregar_dados_dashboard()
    return render_template('dashboard.html',
                           professores=dados_dash.get('professores', []),
                           disciplinas=dados_dash.get('disciplinas', []))

@app.route('/executar', methods=['GET', 'POST'])
def executar():
    """Tela para configurar e executar os algoritmos."""
    if request.method == 'POST':
        # --- Lógica para Processar o Formulário ---
        algoritmos_selecionados = request.form.getlist('algoritmos') # Pega ['pli', 'aco', 'ag']
        resultados_execucao = [] # Lista para guardar os resultados

        # Alocações Fixas (tratamento simples via JSON)
        alocacoes_fixas_str = request.form.get('alocacoes_fixas', '[]')
        try:
            alocacoes_fixas = json.loads(alocacoes_fixas_str)
            if not isinstance(alocacoes_fixas, list): raise ValueError("Deve ser uma lista.")
        except json.JSONDecodeError:
            flash("Formato inválido para Alocações Fixas. Use uma lista de dicionários JSON.", "danger")
            alocacoes_fixas = []
        except ValueError as e:
             flash(f"Erro nas Alocações Fixas: {e}", "danger")
             alocacoes_fixas = []


        # --- Executa cada algoritmo selecionado ---
        for alg_id in algoritmos_selecionados:
            config_atual = None
            OtimizadorClasse = None
            nome_algoritmo = "Desconhecido"

            try:
                if alg_id == 'pli':
                    nome_algoritmo = "PLI"
                    OtimizadorClasse = otimizador_pli.OtimizadorPLI
                    config_atual = config.CONFIG_PLI_PADRAO.copy()
                    config_atual['PENALIDADE_W'] = float(request.form.get('pli_w', config.CONFIG_PLI_PADRAO['PENALIDADE_W']))
                elif alg_id == 'aco':
                    nome_algoritmo = "ACO"
                    OtimizadorClasse = otimizador_aco.OtimizadorACO
                    config_atual = config.CONFIG_ACO_PADRAO.copy()
                    config_atual['ACO_PARAMS'] = {
                        "n_formigas": int(request.form.get('aco_n_formigas', config.CONFIG_ACO_PADRAO['ACO_PARAMS']['n_formigas'])),
                        "n_geracoes": int(request.form.get('aco_n_geracoes', config.CONFIG_ACO_PADRAO['ACO_PARAMS']['n_geracoes'])),
                        "alfa": float(request.form.get('aco_alfa', config.CONFIG_ACO_PADRAO['ACO_PARAMS']['alfa'])),
                        "beta": float(request.form.get('aco_beta', config.CONFIG_ACO_PADRAO['ACO_PARAMS']['beta'])),
                        "taxa_evaporacao": float(request.form.get('aco_taxa_evaporacao', config.CONFIG_ACO_PADRAO['ACO_PARAMS']['taxa_evaporacao'])),
                    }
                elif alg_id == 'ag':
                    nome_algoritmo = "AG"
                    OtimizadorClasse = otimizador_ag.OtimizadorAG
                    config_atual = config.CONFIG_AG_PADRAO.copy()
                    config_atual['AG_PARAMS'] = {
                        "n_populacao": int(request.form.get('ag_n_populacao', config.CONFIG_AG_PADRAO['AG_PARAMS']['n_populacao'])),
                        "n_geracoes": int(request.form.get('ag_n_geracoes', config.CONFIG_AG_PADRAO['AG_PARAMS']['n_geracoes'])),
                        "taxa_crossover": float(request.form.get('ag_taxa_crossover', config.CONFIG_AG_PADRAO['AG_PARAMS']['taxa_crossover'])),
                        "taxa_mutacao": float(request.form.get('ag_taxa_mutacao', config.CONFIG_AG_PADRAO['AG_PARAMS']['taxa_mutacao'])),
                        "tamanho_torneio": int(request.form.get('ag_tamanho_torneio', config.CONFIG_AG_PADRAO['AG_PARAMS']['tamanho_torneio'])),
                        "fator_penalidade": int(request.form.get('ag_fator_penalidade', config.CONFIG_AG_PADRAO['AG_PARAMS']['fator_penalidade'])),
                    }

                # Seed global se informado no formulário
                seed_global = request.form.get('seed')
                if seed_global:
                    try:
                        config_atual['SEED'] = int(seed_global)
                    except Exception:
                        flash('Seed inválida (deve ser inteiro).', 'warning')

                if OtimizadorClasse and config_atual:
                    # Validação antes de executar
                    ok_val, erros_val, avisos_val = validacao.validar_config(alg_id, config_atual)
                    for av in avisos_val:
                        flash(f'Aviso {nome_algoritmo}: {av}', 'info')
                    if not ok_val:
                        for er in erros_val:
                            flash(f'Erro {nome_algoritmo}: {er}', 'danger')
                        continue  # pula execução deste algoritmo
                    config_atual['ALOCACOES_FIXAS'] = alocacoes_fixas # Adiciona fixações
                    print(f"Executando {nome_algoritmo} com config: {config_atual}") # Debug
                    otimizador = OtimizadorClasse(config=config_atual)

                    start_time = time.time()
                    resultado = otimizador.resolver()
                    end_time = time.time()
                    tempo = end_time - start_time

                    if resultado:
                        # Simplifica o resultado para a sessão (DataFrame não é serializável)
                        resultado_simplificado = {
                            'nome_algoritmo': nome_algoritmo,
                            'tempo': tempo,
                            'valor_objetivo': resultado.get('valor_objetivo'),
                            'alocacao': resultado.get('alocacao_final', pd.DataFrame()).to_dict(orient='records'),
                            'soma_preferencias': resultado.get('soma_preferencias'),
                            'num_alocacoes_preferencia_zero': resultado.get('num_alocacoes_preferencia_zero'),
                            'penalidade_total': resultado.get('penalidade_total'),
                            'metricas_iteracao': resultado.get('metricas_iteracao', []),
                            'seed': resultado.get('seed')
                            # Adicione outras métricas se o resolver retornar
                        }
                        # Analisa distribuição de preferências e escore da solução recomposta
                        try:
                            df_aloc = pd.DataFrame(resultado_simplificado['alocacao'])
                            metricas_solucao = analisador_global.avaliar(df_aloc)
                            resultado_simplificado['analise'] = metricas_solucao
                        except Exception as e:
                            print(f"Falha ao analisar solução {nome_algoritmo}: {e}")
                        resultados_execucao.append(resultado_simplificado)
                        # Persistência CSV
                        try:
                            persistencia.salvar_resultado(
                                resultado_simplificado,
                                {"algoritmo": nome_algoritmo, "config": config_atual}
                            )
                        except Exception as e:
                            flash(f"Falha ao salvar histórico: {e}", "warning")
                        flash(f"{nome_algoritmo} executado com sucesso em {tempo:.2f} segundos.", "success")
                    else:
                         flash(f"{nome_algoritmo} não encontrou uma solução viável.", "warning")

            except Exception as e:
                flash(f"Erro ao executar {nome_algoritmo}: {e}", "danger")
                print(f"Erro detalhado ({nome_algoritmo}):", e) # Debug no console

        # Armazena os resultados na sessão para mostrar na próxima tela
        session['resultados'] = resultados_execucao
        return redirect(url_for('resultados'))

    # Se for GET, apenas mostra o formulário
    dados_dash = carregar_dados_dashboard()
    return render_template('executar.html',
                           config_pli=config.CONFIG_PLI_PADRAO,
                           config_aco=config.CONFIG_ACO_PADRAO,
                           config_ag=config.CONFIG_AG_PADRAO,
                           professores=dados_dash.get('professores', []),
                           disciplinas=dados_dash.get('disciplinas', []))

@app.route('/resultados')
def resultados():
    """Tela para exibir os resultados da última execução."""
    # Pega os resultados da sessão (ou uma lista vazia se não houver)
    resultados_da_sessao = session.get('resultados', [])

    # Garante que cada resultado tenha análise (caso venha de sessão antiga sem analise)
    resultados_analisados = []
    for res in resultados_da_sessao:
        if 'analise' not in res:
            try:
                df_aloc = pd.DataFrame(res.get('alocacao', []))
                if not df_aloc.empty:
                    res['analise'] = analisador_global.avaliar(df_aloc)
            except Exception as e:
                print(f"Falha ao analisar resultado em /resultados: {e}")
        resultados_analisados.append(res)
    return render_template('resultados.html', resultados=resultados_analisados)

@app.route('/historico')
def historico():
    """Lista histórico de execuções com filtros simples por algoritmo e seed."""
    algoritmo_filtro = request.args.get('algoritmo')
    seed_filtro = request.args.get('seed')
    historico_raw = persistencia.listar_historico()

    # Processa linhas (parse de alguns campos JSON leves)
    linhas = []
    algoritmos_disponiveis = set()
    for row in historico_raw:
        alg = row.get('algoritmo')
        algoritmos_disponiveis.add(alg)
        if algoritmo_filtro and alg != algoritmo_filtro:
            continue
        if seed_filtro and row.get('seed') != seed_filtro:
            continue
        # Adiciona data legível
        try:
            ts = int(row.get('timestamp', 0))
            from datetime import datetime
            row['data_hora'] = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            row['data_hora'] = row.get('timestamp')
        # Comprimento de métricas
        try:
            import json
            metricas_iter = json.loads(row.get('metricas_iteracao_json', '[]'))
            row['n_iteracoes'] = len(metricas_iter)
        except Exception:
            row['n_iteracoes'] = 0
        linhas.append(row)

    return render_template('historico.html',
                           execucoes=linhas,
                           algoritmos=sorted(list(algoritmos_disponiveis)),
                           algoritmo_filtro=algoritmo_filtro,
                           seed_filtro=seed_filtro)

@app.route('/historico_export')
def historico_export():
    """Exporta em CSV as linhas do histórico aplicando mesmos filtros da tela."""
    algoritmo_filtro = request.args.get('algoritmo')
    seed_filtro = request.args.get('seed')
    historico_raw = persistencia.listar_historico()
    export_rows = []
    for row in historico_raw:
        if algoritmo_filtro and row.get('algoritmo') != algoritmo_filtro:
            continue
        if seed_filtro and row.get('seed') != seed_filtro:
            continue
        export_rows.append(row)
    # Monta CSV em memória
    import io, csv
    output = io.StringIO()
    if export_rows:
        fieldnames = list(export_rows[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)
    else:
        output.write('Nenhum registro selecionado para exportação\n')
    resp = Response(output.getvalue(), mimetype='text/csv')
    nome = 'historico_filtrado.csv'
    resp.headers['Content-Disposition'] = f'attachment; filename={nome}'
    return resp

@app.route('/historico/<id_execucao>')
def historico_detalhe(id_execucao):
    """Mostra detalhes de uma execução específica (alocação e métricas básicas)."""
    # Carrega alocação
    alocacao = persistencia.carregar_alocacao(id_execucao)
    # Busca linha no histórico
    linha_hist = None
    for row in persistencia.listar_historico():
        if row.get('id_execucao') == id_execucao:
            linha_hist = row
            break
    if not linha_hist:
        flash('Execução não encontrada.', 'warning')
        return redirect(url_for('historico'))
    # Parse config
    import json
    try:
        config_json = json.loads(linha_hist.get('config_json', '{}'))
    except Exception:
        config_json = {}
    try:
        metricas_iter = json.loads(linha_hist.get('metricas_iteracao_json', '[]'))
    except Exception:
        metricas_iter = []
    # Analisa solução histórica (se houver alocação)
    analise_historica = None
    try:
        df_hist = pd.DataFrame(alocacao)
        if not df_hist.empty:
            analise_historica = analisador_global.avaliar(df_hist)
    except Exception as e:
        print(f"Falha na análise histórica {id_execucao}: {e}")

    # Enriquecimento: nomes, flag fixada e classificação de preferência
    resumo_preferencias = None
    try:
        df_docentes = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['professores'])
        df_disciplinas = pd.read_csv(config.CONFIG_BASE_ARQUIVOS['disciplinas'])
        map_docentes = df_docentes.set_index('id_docente')['docente'].to_dict()
        map_disc = df_disciplinas.set_index('id_disciplina')['disciplina'].to_dict()

        alocacoes_fixas_cfg = set()
        try:
            fix_list = config_json.get('config', {}).get('ALOCACOES_FIXAS', []) or config_json.get('ALOCACOES_FIXAS', [])
            for fx in fix_list:
                alocacoes_fixas_cfg.add((str(fx.get('id_docente')).strip(), str(fx.get('id_disciplina')).strip()))
        except Exception:
            pass

        def classificar(pref, fixada=False):
            if fixada:
                return 'fixada', 'pref-fixada', 'Fixada'
            if pref == 3:
                return 'alta', 'pref-alta', 'Alto'
            if pref == 2:
                return 'media', 'pref-media', 'Médio'
            if pref == 1:
                return 'baixa', 'pref-baixa', 'Baixo'
            return 'baixa', 'pref-baixa', 'Baixo'

        contagem = {'pref_3':0,'pref_2':0,'pref_1':0,'pref_0':0}
        total_alocs = len(alocacao)
        for linha in alocacao:
            doc = linha.get('id_docente')
            disc = linha.get('id_disciplina')
            linha['docente_nome'] = map_docentes.get(doc, doc)
            linha['disciplina_nome'] = map_disc.get(disc, disc)
            try:
                pref_val = int(linha.get('preferencia'))
            except Exception:
                try:
                    pref_val = analisador_global.dados_preparados['preferencias'][doc][disc]
                except Exception:
                    pref_val = -1
                linha['preferencia'] = pref_val
            fixada = (doc, disc) in alocacoes_fixas_cfg
            nivel_slug, nivel_css, nivel_label = classificar(pref_val, fixada)
            linha['nivel_preferencia'] = nivel_slug
            linha['nivel_css'] = nivel_css
            linha['nivel_label'] = nivel_label
            linha['fixada'] = fixada
            if pref_val in [0,1,2,3]:
                contagem[f'pref_{pref_val}'] += 1
        if total_alocs > 0:
            resumo_preferencias = {
                'total': total_alocs,
                'pref_3': contagem['pref_3'],
                'pref_2': contagem['pref_2'],
                'pref_1': contagem['pref_1'],
                'pref_0': contagem['pref_0'],
                'pct_3': (contagem['pref_3']/total_alocs)*100,
                'pct_2': (contagem['pref_2']/total_alocs)*100,
                'pct_1': (contagem['pref_1']/total_alocs)*100,
                'pct_0': (contagem['pref_0']/total_alocs)*100,
            }
    except Exception as e:
        print(f"Falha ao enriquecer histórico {id_execucao}: {e}")

    return render_template('historico_detalhe.html',
                           id_execucao=id_execucao,
                           dados=linha_hist,
                           alocacao=alocacao,
                           config=config_json,
                           metricas_iteracao=metricas_iter,
                           analise=analise_historica,
                           resumo_preferencias=resumo_preferencias)

@app.route('/historico/<id_execucao>/export_alocacao')
def export_alocacao(id_execucao):
    """Exporta a alocação final de uma execução específica em CSV."""
    registros = persistencia.carregar_alocacao(id_execucao)
    import io, csv
    output = io.StringIO()
    if registros:
        fieldnames = sorted({k for r in registros for k in r.keys()})
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(registros)
    else:
        output.write('Sem alocação para este id_execucao\n')
    resp = Response(output.getvalue(), mimetype='text/csv')
    resp.headers['Content-Disposition'] = f'attachment; filename=alocacao_{id_execucao}.csv'
    return resp

@app.route('/executar_async', methods=['POST'])
def executar_async():
    """Dispara execução assíncrona de um algoritmo e retorna job_id."""
    dados_req = request.get_json(force=True)
    alg_id = dados_req.get('algoritmo')
    if not alg_id:
        return {'erro': 'algoritmo é obrigatório'}, 400
    # Monta config semelhante ao formulário (apenas parâmetros relevantes passados)
    config_base = {
        'ARQUIVOS_DADOS': config.CONFIG_BASE_ARQUIVOS,
        'ALOCACOES_FIXAS': dados_req.get('alocacoes_fixas', []),
    }
    # Merge especifico
    if alg_id == 'pli':
        cfg = config.CONFIG_PLI_PADRAO.copy()
        w = dados_req.get('pli_w')
        if w is not None:
            cfg['PENALIDADE_W'] = float(w)
        config_final = {**config_base, **cfg}
    elif alg_id == 'aco':
        cfg = config.CONFIG_ACO_PADRAO.copy()
        params = cfg['ACO_PARAMS'].copy()
        for k in ['n_formigas','n_geracoes','alfa','beta','taxa_evaporacao']:
            if k in dados_req:
                params[k] = type(params[k])(dados_req[k])
        cfg['ACO_PARAMS'] = params
        config_final = {**config_base, **cfg}
    elif alg_id == 'ag':
        cfg = config.CONFIG_AG_PADRAO.copy()
        params = cfg['AG_PARAMS'].copy()
        for k in ['n_populacao','n_geracoes','taxa_crossover','taxa_mutacao','tamanho_torneio','fator_penalidade']:
            if k in dados_req:
                params[k] = type(params[k])(dados_req[k])
        cfg['AG_PARAMS'] = params
        config_final = {**config_base, **cfg}
    else:
        return {'erro': f'Algoritmo desconhecido: {alg_id}'}, 400

    seed = dados_req.get('seed')
    if seed is not None:
        config_final['SEED'] = seed

    # Validação
    ok_val, erros_val, avisos_val = validacao.validar_config(alg_id, config_final)
    if not ok_val:
        return {'erro': ' ; '.join(erros_val)}, 400
    # Opcional: retornar avisos ao cliente (não bloqueia)
    if avisos_val:
        print(f'Avisos validação {alg_id}: {avisos_val}')

    job_id = tarefas.iniciar_job(alg_id, config_final)
    return {'job_id': job_id}

@app.route('/progresso/<job_id>')
def progresso(job_id):
    estado = tarefas.obter_estado(job_id)
    if not estado:
        return {'erro': 'Job não encontrado'}, 404
    return estado

@app.route('/simulacoes', methods=['GET','POST'])
def simulacoes():
    """Formulário para criar lote e painel para acompanhar."""
    if request.method == 'POST':
        # Coleta parâmetros de ranges e seeds
        algoritmos = request.form.getlist('algoritmos')
        seeds_str = request.form.get('seeds','').strip()
        seeds = [s for s in seeds_str.split(',') if s.strip()]
        def build_range(prefix):
            try:
                start = float(request.form.get(f'{prefix}_start'))
                stop = float(request.form.get(f'{prefix}_stop'))
                step = float(request.form.get(f'{prefix}_step'))
                values = []
                v = start
                # Proteção para step zero
                if step == 0:
                    return [start]
                while (step > 0 and v <= stop) or (step < 0 and v >= stop):
                    values.append(v)
                    v += step
                return values
            except (TypeError, ValueError):
                return []

        def parse_list(name):
            raw = request.form.get(name,'').strip()
            if not raw:
                return []
            out = []
            for part in raw.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    out.append(float(part))
                except ValueError:
                    flash(f'Valor inválido em {name}: {part}', 'warning')
            return out

        itens = []
        # Gera definições por algoritmo variando um parâmetro principal + seeds
        if 'pli' in algoritmos:
            w_vals = build_range('pli_w') or [request.form.get('pli_w_start', config.CONFIG_PLI_PADRAO['PENALIDADE_W'])]
            for w in w_vals:
                for sd in seeds or [None]:
                    cfg = config.CONFIG_PLI_PADRAO.copy()
                    cfg['PENALIDADE_W'] = float(w)
                    if sd is not None: cfg['SEED'] = int(sd)
                    cfg['ARQUIVOS_DADOS'] = config.CONFIG_BASE_ARQUIVOS
                    itens.append({'alg_id':'pli','config':cfg})
        if 'aco' in algoritmos:
            gen_vals = build_range('aco_n_geracoes') or [config.CONFIG_ACO_PADRAO['ACO_PARAMS']['n_geracoes']]
            alfa_list = parse_list('aco_alfa_list') or [config.CONFIG_ACO_PADRAO['ACO_PARAMS']['alfa']]
            beta_list = parse_list('aco_beta_list') or [config.CONFIG_ACO_PADRAO['ACO_PARAMS']['beta']]
            for g in gen_vals:
                for alfa in alfa_list:
                    for beta in beta_list:
                        for sd in seeds or [None]:
                            cfg = config.CONFIG_ACO_PADRAO.copy()
                            params = cfg['ACO_PARAMS'].copy()
                            params['n_geracoes'] = int(g)
                            params['alfa'] = float(alfa)
                            params['beta'] = float(beta)
                            cfg['ACO_PARAMS'] = params
                            if sd is not None: cfg['SEED'] = int(sd)
                            cfg['ARQUIVOS_DADOS'] = config.CONFIG_BASE_ARQUIVOS
                            itens.append({'alg_id':'aco','config':cfg})
        if 'ag' in algoritmos:
            gen_vals = build_range('ag_n_geracoes') or [config.CONFIG_AG_PADRAO['AG_PARAMS']['n_geracoes']]
            mut_list = parse_list('ag_taxa_mutacao_list') or [config.CONFIG_AG_PADRAO['AG_PARAMS']['taxa_mutacao']]
            for g in gen_vals:
                for mut in mut_list:
                    for sd in seeds or [None]:
                        cfg = config.CONFIG_AG_PADRAO.copy()
                        params = cfg['AG_PARAMS'].copy()
                        params['n_geracoes'] = int(g)
                        params['taxa_mutacao'] = float(mut)
                        cfg['AG_PARAMS'] = params
                        if sd is not None: cfg['SEED'] = int(sd)
                        cfg['ARQUIVOS_DADOS'] = config.CONFIG_BASE_ARQUIVOS
                        itens.append({'alg_id':'ag','config':cfg})
        if not itens:
            flash('Nenhum item gerado para o lote. Verifique parâmetros.', 'warning')
            return redirect(url_for('simulacoes'))
        # Valida cada item antes de iniciar
        ok_batch, errs_batch, avis_batch = validacao.validar_batch_items(itens)
        for av in avis_batch:
            flash(f'Aviso lote: {av}', 'info')
        if not ok_batch:
            for er in errs_batch:
                flash(f'Erro lote: {er}', 'danger')
            return redirect(url_for('simulacoes'))
        group_id = tarefas.iniciar_batch({'items': itens, 'meta': {'seeds': seeds}})
        return redirect(url_for('simulacoes_resultados', group_id=group_id))

    return render_template('simulacoes.html')

@app.route('/simulacoes_progresso/<group_id>')
def simulacoes_progresso(group_id):
    estado = tarefas.obter_estado_grupo(group_id)
    if not estado:
        return {'erro':'Grupo não encontrado'}, 404
    return estado

@app.route('/simulacoes_resultados/<group_id>')
def simulacoes_resultados(group_id):
    estado = tarefas.obter_estado_grupo(group_id)
    if not estado:
        flash('Grupo não encontrado','warning')
        return redirect(url_for('simulacoes'))
    # Se concluído, agrega estatísticas a partir do histórico
    agregado = None
    if estado['status'] == 'done':
        import json
        hist = persistencia.listar_historico()
        valores = []
        tempos = []
        distribs = []
        for row in hist:
            try:
                cfg = json.loads(row.get('config_json','{}'))
                if cfg.get('config', {}).get('GROUP_ID') == group_id:
                    if row.get('valor_objetivo'): valores.append(float(row['valor_objetivo']))
                    if row.get('tempo_execucao'): tempos.append(float(row['tempo_execucao']))
                    # distrib de preferencias não salva; podemos inferir depois lendo alocação se quiser.
            except Exception:
                continue
        if valores:
            import statistics
            agregado = {
                'qtd': len(valores),
                'media_objetivo': statistics.mean(valores),
                'melhor': max(valores),
                'pior': min(valores),
                'desvio_objetivo': statistics.pstdev(valores) if len(valores)>1 else 0,
                'media_tempo': statistics.mean(tempos) if tempos else None
            }
    return render_template('simulacoes_resultados.html', group=estado, agregado=agregado)

@app.route('/simulacoes_resultados/<group_id>/export')
def simulacoes_resultados_export(group_id):
    """Exporta CSV com execuções do grupo e métricas agregadas."""
    import json, io, csv
    hist = persistencia.listar_historico()
    linhas = []
    for row in hist:
        try:
            cfg = json.loads(row.get('config_json','{}'))
            if cfg.get('config', {}).get('GROUP_ID') == group_id:
                linhas.append(row)
        except Exception:
            continue
    output = io.StringIO()
    if linhas:
        fieldnames = list(linhas[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(linhas)
        # Estatísticas simples
        valores = [float(l['valor_objetivo']) for l in linhas if l.get('valor_objetivo')]
        if valores:
            import statistics
            stats_row = {
                k: '' for k in fieldnames
            }
            stats_row['algoritmo'] = 'AGREGADO'
            stats_row['valor_objetivo'] = f"media={statistics.mean(valores):.4f};melhor={max(valores):.4f};pior={min(valores):.4f};desvio={statistics.pstdev(valores) if len(valores)>1 else 0:.4f}"
            writer.writerow(stats_row)
    else:
        output.write('Nenhuma execução encontrada para o grupo\n')
    resp = Response(output.getvalue(), mimetype='text/csv')
    resp.headers['Content-Disposition'] = f'attachment; filename=grupo_{group_id}.csv'
    return resp


if __name__ == '__main__':
    # Garante que a pasta 'templates' existe
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Adiciona a pasta do projeto ao path para garantir que Flask encontre os módulos
    # Já garantido no topo, mas reforça caso execução direta fora de FLASK_APP
    if str(PROJETO_APLICADO_DIR) not in sys.path:
        sys.path.insert(0, str(PROJETO_APLICADO_DIR))
    # Inicializa camada de persistência CSV
    try:
        persistencia.inicializar()
    except Exception as e:
        print(f"Não foi possível inicializar persistência CSV: {e}")

    app.run(debug=True)