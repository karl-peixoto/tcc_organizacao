import os
import sys
import time
import pandas as pd
import multiprocessing as mp

import numpy as np
import numpy.random as rng
import re
from functools import wraps
from contextlib import contextmanager
import contextlib
from datetime import datetime

# Inserir caminho do projeto para imports locais
PROJ_ROOT = r"C:\Users\kmenezes\OneDrive - unb.br\tcc_organizacao"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from projeto_aplicado.modelos.otimizador_pli import OtimizadorPLI
from projeto_aplicado.modelos.analisador import AnalisadorDeSolucao
from projeto_aplicado.modelos.otimizador_aco import OtimizadorACO
from projeto_aplicado.modelos.otimizador_ag import OtimizadorAG
from projeto_aplicado.modelos.otimizador_base import Otimizador



@contextmanager
def suppress_stdout():
    """Um context manager que redireciona o stdout para devnull."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def silence_output(func):
    """Decorator para suprimir prints de uma função."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with suppress_stdout():
            return func(*args, **kwargs)
    return wrapper

CONFIG_BASE = {
    "ARQUIVOS_DADOS": {
        "disciplinas": "disciplinas.csv",
        "professores": "docentes.csv",
        "preferencias": "preferencias.csv",
        "conflitos": "matriz_conflitos.csv" 
    },
    "ALOCACOES_FIXAS": [
    ]
}
CONFIG_BASE_ADAPTADA = {
    "ALOCACOES_FIXAS": [
    ]
}

# Controle de execução por algoritmo
ALGORITMOS_ATIVOS = {
    'PLI': True,
    'ACO': True,
    'AG': True,
}

# Parâmetros específicos para cada algoritmo
CONFIG_PLI = {
    **CONFIG_BASE,
    "PENALIDADE_W": 5.0
}

CONFIG_ACO = {
    **CONFIG_BASE,
    "ACO_PARAMS": {
        "n_geracoes": 100,
    }
}

CONFIG_AG = {
    **CONFIG_BASE,
    "AG_PARAMS": {
        "n_geracoes": 350,
        "fator_penalidade": 10,
    }
}

analise = AnalisadorDeSolucao(config=CONFIG_BASE)
otimizador = Otimizador(config=CONFIG_BASE)
d1 = otimizador.dados_brutos
prof = d1['professores']
disc = d1['disciplinas']
conf = d1['conflitos']
pref = d1['preferencias']

seed = 1 #Seed global do estudo das simulacoes
seed = 211028972 #Seed utilizada na validação de robustez

def log_msg(mensagem):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {mensagem}")

#Criando matriz de conflitos
def parse_horario(horario_str):
    """
    Interpreta uma string de horário no formato '24M12' e a converte
    em um dicionário estruturado.

    Args:
        horario_str (str): A string de horário (ex: '35T34', '6N12', etc.).

    Returns:
        dict: Um dicionário com os dias, turno e horários, ou None se o formato for inválido.
    """
    # Expressão regular para capturar os 3 componentes: dias, turno, blocos
    match = re.match(r'(\d+)([MTN])(\d+)', horario_str)
    
    if not match:
        return None # Formato inválido

    dias_str, turno_char, blocos_str = match.groups()

    mapa_dias = {'2': 'SEG', '3': 'TER', '4': 'QUA', '5': 'QUI', '6': 'SEX'}
    
    # Converte os dígitos dos dias em um conjunto de strings (ex: {'SEG', 'QUA'})
    dias_set = {mapa_dias[dia] for dia in dias_str if dia in mapa_dias}
    
    # Converte os dígitos dos blocos em um conjunto de inteiros (ex: {1, 2})
    blocos_set = {int(bloco) for bloco in blocos_str}

    return {
        'dias': dias_set,
        'turno': turno_char,
        'blocos': blocos_set
    }

def verificar_conflito(horario_str1, horario_str2):
    """
    Verifica se duas strings de horário entram em conflito.

    Args:
        horario_str1 (str): Horário da primeira disciplina.
        horario_str2 (str): Horário da segunda disciplina.

    Returns:
        bool: True se houver conflito, False caso contrário.
    """
    h1 = parse_horario(horario_str1)
    h2 = parse_horario(horario_str2)

    if h1 is None or h2 is None:
        print(f"Aviso: Formato de horário inválido encontrado ({horario_str1} ou {horario_str2})")
        return False

    # 1. Verifica se os turnos são diferentes (se forem, não há conflito)
    if h1['turno'] != h2['turno']:
        return False

    # 2. Verifica se há intersecção de dias da semana
    dias_em_comum = h1['dias'].intersection(h2['dias'])
    if not dias_em_comum:
        return False

    # 3. Verifica se há intersecção de blocos de horário
    blocos_em_comum = h1['blocos'].intersection(h2['blocos'])
    if not blocos_em_comum:
        return False

    # Se passou por todas as verificações, significa que há conflito
    return True


def criar_matriz_conflitos(disciplinas_df):
    """
    Cria uma matriz de conflitos entre disciplinas com base nos horários.
    
    Args:
        disciplinas_df (pd.DataFrame): DataFrame com colunas 'id_disciplina' e 'horario'
    
    Returns:
        pd.DataFrame: Matriz de conflitos (0 = sem conflito, 1 = com conflito)
    """
    disciplinas = disciplinas_df['id_disciplina'].tolist()
    horarios = disciplinas_df.set_index('id_disciplina')['horario'].to_dict()
    num_disciplinas = len(disciplinas)
    matriz_conflitos = pd.DataFrame(0, index=disciplinas, columns=disciplinas)
    
    for i in range(num_disciplinas):
        for j in range(i, num_disciplinas):
            disc1 = disciplinas[i]
            disc2 = disciplinas[j]
            
            # Não compara uma disciplina com ela mesma
            if i == j:
                continue
                
            if verificar_conflito(horarios[disc1], horarios[disc2]):
                matriz_conflitos.loc[disc1, disc2] = 1
                matriz_conflitos.loc[disc2, disc1] = 1
    
    return matriz_conflitos

def normalizar_probabilidades(p):
    """
    Ajusta um vetor de probabilidades para ser válido:
    - todos positivos
    - soma exatamente 1
    metodo:
      - "clip": zera negativos e normaliza
    """
    p = np.asarray(p, dtype=float)


    # Estável numericamente
    p = np.clip(p, 0, None)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))

#Função que perturba preferencias
def perturbar_preferencias(df_preferencias, df_disciplinas,  seed=None):
    """
    Altera as preferencias forma aleatória.
    
    Parameters:
    -----------
    df_preferencias : pd.DataFrame
        DataFrame contendo pelo menos a coluna 'preferencia'
    proporcao_perturbacao : float
        Proporção de preferências a serem alteradas (0 a 1)
    seed : int, optional
        Seed para reprodutibilidade
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com as preferências perturbadas
    """
    if seed is not None:
        rng.seed(seed)
    
    #Calcula o número de turmas por disciplina
    turmas_disc = df_disciplinas[['id_disciplina', 'disciplina']]\
        .merge(df_disciplinas.groupby('disciplina').agg({'id_disciplina':'count'}).rename(columns={'id_disciplina':'n_turmas'}).reset_index(), on='disciplina', how='left')\
            .copy()
    
    #Cria o dataframe que será alterado
    df_result = df_preferencias.copy()
    #media_sensibilidade = .1
    #sd_sensibilidade = .09
    media_robustez = .3
    sd_robustez = .15
    proporcao_perturbacao = min(max(rng.normal(media_robustez, sd_robustez), 0), 1)
    opcoes = [0, 1, 2, 3]
    prob_original = [0.43, 0.21, 0.07, 0.29]
    variabilidade_sensibilidade = 0.05
    variabilidade_robustez = 0.1
    probabilidades = normalizar_probabilidades(prob_original + rng.normal(loc=0, scale=variabilidade_robustez, size=4))
    n_alteracoes = int(len(df_result) * proporcao_perturbacao)
    
    try:
        escolhas = rng.choice(opcoes, 
                          size=n_alteracoes,
                          replace=True,
                          p=probabilidades)
    except:
        escolhas = rng.choice(opcoes, 
                          size=n_alteracoes,
                          replace=True,
                          p=prob_original)
    
    indices_aleatorios = df_result.sample(n = n_alteracoes).index
    df_result.loc[indices_aleatorios, 'preferencia'] = escolhas
    
    #Calcula métricas de variacao de preferências
    #Calcula numero de professores interessados por turma
    analise = df_result[df_result['preferencia'] ==3].groupby('id_disciplina').agg({'id_docente':'count'}).reset_index().rename(columns={'id_docente':'n_professores_interessados'})
    #Adiciona numero de turmas por disciplina
    analise = analise.merge(turmas_disc, on='id_disciplina', how='outer')
    #Calcula a media de professores interessados por disciplina
    analise = analise.groupby(['disciplina']).agg({'id_disciplina':'first', 'n_professores_interessados':'mean', 'n_turmas':'first'}).reset_index()
    #Calcula interessados por turma
    analise['interesados_por_turma'] = analise['n_professores_interessados'] / analise['n_turmas']
    #Calcula global de interessados por turma
    n_interessados_por_turma = analise['interesados_por_turma'].mean()
    metricas = {
        'proporcao_perturbacao': proporcao_perturbacao,
        'n_interessados_por_turma': n_interessados_por_turma
    }

    return metricas, df_result


def perturbar_horarios(df_disciplinas, seed=None):
    """
    Perturba os horários das disciplinas selecionando aleatoriamente novas opções 
    com base na distribuição de frequência existente para cada par (Carga Horária, Tipo).
    
    A função:
    1. Calcula a distribuição de probabilidade dos horários atuais.
    2. Sorteia uma proporção variável de disciplinas para alterar.
    3. Para cada disciplina selecionada, escolhe um novo horário compatível, 
       adicionando um ruído estocástico às probabilidades originais para garantir variabilidade.
    4. Recalcula a matriz de conflitos e retorna o saldo (delta) em relação ao estado original.

    Parameters:
    -----------
    df_disciplinas : pd.DataFrame
        DataFrame contendo as disciplinas com colunas 'horario', 'carga_horaria' e 'tipo_disciplina'.
    seed : int, optional
        Seed para reprodutibilidade do gerador de números aleatórios.
        
    Returns:
    --------
    tuple
        (metricas: dict, matriz_conflitos_perturbada: pd.DataFrame)
        Onde 'metricas' contém o número de mudanças e o saldo de conflitos gerados/resolvidos.
    """
    if seed is not None:
        rng.seed(seed)
    
    df_result = df_disciplinas.copy()
    #Calcula as probabilidades de cada horario baseado na carga horaria e tipo de disciplina
    probs = df_result[['carga_horaria', 'tipo_disciplina', 'horario']].value_counts(normalize=True)
    media_sensibilidade = .2
    sd_sensibilidade = .05
    media_robustez = .4
    sd_robustez = .1
    #Seleciona aleatoriamente uma proporção de disciplinas para alterar o horario
    proporcao_alteracoes = min(max(rng.normal(media_robustez, sd_robustez, 1 ), 0), 1)
    indices_escolhidos = df_result.sample(frac=float(proporcao_alteracoes)).index
    
    for indice in indices_escolhidos:
        try:
            i = df_result.loc[indice]
            #Seleciona as opcoes possiveis para o tipo de disciplina e carga horaria disponivel
            opcoes = probs[i['carga_horaria']][i['tipo_disciplina']]
            #Calcula as probabilidades de cada horario e adiciona um ruido normal
            #sd_sensibilidade = 0.009
            sd_robustez = 0.01
            probabilidades = normalizar_probabilidades(opcoes.values + rng.normal(loc=0, scale=sd_robustez, size=len(opcoes.values)))
            #Escolhe um novo horario baseado nas probabilidades
            df_result.at[indice, 'horario'] = rng.choice(opcoes.index, size=1, p = probabilidades)[0]
        except:
            continue    
    #Nº de conflitos na matriz original é 146
    n_mudancas_horarios = len(indices_escolhidos)
    matriz_conflitos = criar_matriz_conflitos(df_result)

    #Saldo de conflitos
    n_conflitos = matriz_conflitos.sum().sum()/2
    metricas = {
        'n_mudancas_horarios': n_mudancas_horarios,
        'n_conflitos': n_conflitos
    }
    return metricas, matriz_conflitos


def perturbar_max_disciplinas(df_professores, seed=None):
    """
    Aumenta aleatoriamente o número máximo de disciplinas para professores que têm max_disciplinas = 1.
    
    Parameters:
    -----------
    df_professores : pd.DataFrame
        DataFrame contendo informações dos professores com coluna 'max_disciplinas'
    seed : int, optional
        Seed para reprodutibilidade
        
    Returns:
    --------
    tuple
        (folga_ganha, df_result) onde:
        - folga_ganha: número de professores que ganharam folga
        - df_result: DataFrame com max_disciplinas atualizado
    """
    if seed is not None:
        rng.seed(seed)
    
    df_result = df_professores.copy()
    folga_ganha = rng.randint(0, 6)
    indices_aleatorios = df_result[df_result['max_disciplinas'] == 1].sample(folga_ganha).index
    df_result.loc[indices_aleatorios, 'max_disciplinas'] += 1
    metricas = {
        'folga_ganha': folga_ganha
    }
    return metricas, df_result


def executar_simulacao(idx, row):
    """
    Executa uma simulação completa com os três algoritmos de otimização.
    
    Parameters:
    -----------
    idx : int
        Índice da simulação
    row : pd.Series
        Linha do DataFrame contendo os parâmetros da simulação        
    Returns:
    --------
    dict
        Dicionário com os resultados da simulação
    """
    inicio_geral = time.process_time()
    
    # Gerar dados perturbados
    metricas_preferencia, pref_perturbado = perturbar_preferencias(pref, disc, seed=seed+idx)
    metricas_horario, conf_perturbado = perturbar_horarios(disc, seed=seed+idx)
    metricas_carga, prof_perturbado = perturbar_max_disciplinas(prof, seed=seed+idx)
    
    tempo_perturbacao = time.process_time() - inicio_geral
    
    # Atualizar dados perturbados
    dados_perturbados = {
        'disciplinas': disc,
        'professores': prof_perturbado,
        'preferencias': pref_perturbado,
        'conflitos': conf_perturbado
    }
    
    # Otimização PLI (condicional)
    resultado_pli = None
    tempo_pli = None
    if ALGORITMOS_ATIVOS.get('PLI', False):
        config_pli_temp = {**CONFIG_PLI}
        config_pli_temp['DADOS_INJETADOS'] = dados_perturbados
        otim_pli = OtimizadorPLI(config=config_pli_temp)
        inicio = time.process_time()
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                resultado_pli = otim_pli.resolver()
        tempo_pli = time.process_time() - inicio
    
    # Otimização ACO
    config_aco_temp = {**CONFIG_ACO}
    config_aco_temp['DADOS_INJETADOS'] = dados_perturbados
    config_aco_temp['ACO_PARAMS']['n_formigas'] = int(row['n_formigas'])
    config_aco_temp['ACO_PARAMS']['alfa'] = row['alfa']
    config_aco_temp['ACO_PARAMS']['beta'] = row['beta']
    config_aco_temp['ACO_PARAMS']['taxa_evaporacao'] = row['taxa_evaporacao']
    
    resultado_aco = None
    tempo_aco = None
    if ALGORITMOS_ATIVOS.get('ACO', False):
        otim_aco = OtimizadorACO(config=config_aco_temp)
        inicio = time.process_time()
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                resultado_aco = otim_aco.resolver()
        tempo_aco = time.process_time() - inicio
   
    # Otimização AG
    config_ag_temp = {**CONFIG_AG}
    config_ag_temp['DADOS_INJETADOS'] = dados_perturbados
    config_ag_temp['AG_PARAMS']['n_populacao'] = int(row['n_populacao'])
    config_ag_temp['AG_PARAMS']['taxa_crossover'] = row['taxa_crossover']
    config_ag_temp['AG_PARAMS']['taxa_mutacao'] = row['taxa_mutacao']
    config_ag_temp['AG_PARAMS']['tamanho_torneio'] = int(row['tamanho_torneio'])
    config_ag_temp['AG_PARAMS']['tamanho_elite'] = int(row['n_populacao'] * row['elit_pct'])
    
    resultado_ag = None
    tempo_ag = None
    if ALGORITMOS_ATIVOS.get('AG', False):
        otim_ag = OtimizadorAG(config=config_ag_temp)
        inicio = time.process_time()
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                resultado_ag = otim_ag.resolver()
        tempo_ag = time.process_time() - inicio

    # Salvar métricas por geração do AG (append no mesmo CSV)

    def _geracao_convergencia_primeira(metricas_iteracao):
        """
        Retorna a primeira geração em que o melhor valor final da função objetivo
        foi alcançado. Não retorna o valor do objetivo, apenas o número da geração.
        """
        try:
            met_df = pd.DataFrame(metricas_iteracao)
            max = met_df['melhor_global'].max()
            pri_ger = int(met_df[met_df['melhor_geracao'] == max]['geracao'].min())
            return pri_ger
        except Exception:
            return None

    def _numero_de_geracoes(metricas_iteracao):
        """
        Retorna o número total de gerações executadas.
        Usa max(geracao) + 1 quando disponível; caso contrário, len(metricas_iteracao).
        """
        try:
            if not isinstance(metricas_iteracao, (list, tuple)) or len(metricas_iteracao) == 0:
                return None
            dfm = pd.DataFrame(metricas_iteracao)
            if 'geracao' in dfm.columns:
                return int(dfm['geracao'].max())
            return len(metricas_iteracao)
        except Exception:
            return None
    
    # Avaliar soluções
    config_analise_temp = {**CONFIG_BASE_ADAPTADA}
    config_analise_temp['DADOS_INJETADOS'] = dados_perturbados
    analise = AnalisadorDeSolucao(config=config_analise_temp)

    log_msg(f"Simulação {idx+1} concluída. Tempo Perturbação: {tempo_perturbacao:.2f}s, PLI: {tempo_pli:.2f}s, ACO: {tempo_aco:.2f}s, AG: {tempo_ag:.2f}s")
    # Log seguro dos resultados
    vo_pli = (resultado_pli.get('valor_objetivo') if isinstance(resultado_pli, dict) else None)
    vo_aco = (resultado_aco.get('valor_objetivo') if isinstance(resultado_aco, dict) else None)
    vo_ag = (resultado_ag.get('valor_objetivo') if isinstance(resultado_ag, dict) else None)
    log_msg(f"Resultados - PLI: {vo_pli} | ACO: {vo_aco} | AG: {vo_ag}")

    # Cálculo de gerações e convergência (primeira geração que atingiu o melhor)
    geracoes_aco_calc = (_numero_de_geracoes(resultado_aco.get('metricas_iteracao')) if isinstance(resultado_aco, dict) else None)
    geracoes_ag_calc = (_numero_de_geracoes(resultado_ag.get('metricas_iteracao')) if isinstance(resultado_ag, dict) else None)
    conv_aco_primeira = (_geracao_convergencia_primeira(resultado_aco.get('metricas_iteracao')) if isinstance(resultado_aco, dict) else None)
    conv_ag_primeira = (_geracao_convergencia_primeira(resultado_ag.get('metricas_iteracao')) if isinstance(resultado_ag, dict) else None)

    return {
        'simulacao': row['simulacao'],
        'tempo_perturbacao':tempo_perturbacao,
        'metricas_preferencia': metricas_preferencia,
        'metricas_horario': metricas_horario,
        'folga_total': metricas_carga['folga_ganha'],
        'tempo_pli': tempo_pli,
        'tempo_aco': tempo_aco,
        'tempo_ag': tempo_ag,
        'geracoes_aco': geracoes_aco_calc,
        'geracoes_ag': geracoes_ag_calc,
        'geracao_convergencia_aco': conv_aco_primeira,
        'geracao_convergencia_ag': conv_ag_primeira,
        'resultado_pli': ([analise.avaliar(resultado_pli['alocacao_final'])] if isinstance(resultado_pli, dict) else None),
        'resultado_aco': ([analise.avaliar(resultado_aco['alocacao_final'])] if isinstance(resultado_aco, dict) else None),
        'resultado_ag': ([analise.avaliar(resultado_ag['alocacao_final'])] if isinstance(resultado_ag, dict) else None)
    }



if __name__ == "__main__":
    lhs_global = pd.read_csv(r"C:\Users\kmenezes\OneDrive - unb.br\tcc_organizacao\codigos\simulacao_robustez.csv")
    # Limitar a 10 simulações
    #lhs_global = lhs_global.head(10)

    print(lhs_global)

    num_batches = 1
    batch_size = len(lhs_global)/num_batches

    caminho = "C:\\Users\\kmenezes\\OneDrive - unb.br\\tcc_organizacao\\codigos"
    num_cores = max(1, mp.cpu_count() - 1) 

    log_msg(f"Iniciando processamento paralelo com {num_cores} núcleos.")
    log_msg(f"Carregado LHS Global com {len(lhs_global)} registros.")


    for i in range(num_batches):
        linf = (i * batch_size)
        lsup = (i+1) * batch_size
        batch = lhs_global[(lhs_global.index + 1 <= lsup) & (lhs_global.index + 1 >linf)]
        log_msg("Resolvendo batch {}/{} com {} simulações".format(i+1, num_batches, len(batch)))

        argumentos = [(idx, row) for idx, row in batch.iterrows()]
        resultados = []
        try:
            # Cria o Pool de processos e executa em paralelo
            with mp.Pool(processes=num_cores) as pool:
                # starmap desempacota a tupla (idx, row) para a função executar_simulacao(idx, row)
                resultados = pool.starmap(executar_simulacao, argumentos)
            
            # Salva os resultados do batch
            final = pd.DataFrame(resultados)
            arquivo_saida = f'{caminho}/dados_robustez.csv'
            #arquivo_saida = os.path.join(caminho, f'estudo_convergencia.csv')
            final.to_csv(arquivo_saida, index=False)
            log_msg(f"Batch {i+1} salvo com sucesso: {arquivo_saida}")
                
        except Exception as e:
            log_msg(f"CRITICAL ERROR no Batch {i+1}: {e}")
            # Opcional: break para parar tudo se der erro, ou continue para tentar o próximo

    log_msg("Todas as simulações foram concluídas.")









