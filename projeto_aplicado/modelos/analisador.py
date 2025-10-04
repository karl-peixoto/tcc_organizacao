import pandas as pd
import numpy as np
from pathlib import Path

class AnalisadorDeSolucao:
    """
    Uma classe autônoma para avaliar a qualidade de qualquer solução de alocação,
    seja ela histórica ou gerada por um otimizador.
    """
    def __init__(self, config: dict):
        """
        Inicializa o analisador carregando e preparando todos os dados de contexto
        necessários para a avaliação.

        Args:
            config (dict): Dicionário de configuração contendo a chave "ARQUIVOS_DADOS"
                           com os nomes dos arquivos de disciplinas, docentes e preferências.
        """
        self.config = config
        self.dados_preparados = None
        self._carregar_e_preparar_dados()
        print("Analisador de Solução autônomo pronto.")

    def _carregar_e_preparar_dados(self):
        """
        Carrega e prepara todos os dados de contexto (professores, disciplinas, preferências)
        para que as soluções possam ser avaliadas.
        (Lógica adaptada da classe Otimizador)
        """
        # Carrega os dados brutos
        try:
            caminho_raiz = Path(__file__).parent.parent.parent
            caminho_pasta_dados = caminho_raiz / "dados"
            dados_brutos = {
                nome: pd.read_csv(caminho_pasta_dados / nome_arquivo)
                for nome, nome_arquivo in self.config["ARQUIVOS_DADOS"].items()
            }
        except FileNotFoundError as e:
            print(f"Erro ao carregar arquivos de dados: {e}")
            raise

        # Prepara os dados (limpeza, padronização, etc.)
        df_professores = dados_brutos["professores"].copy()
        df_professores['id_docente'] = df_professores['id_docente'].str.strip()
        
        df_disciplinas = dados_brutos["disciplinas"].copy()
        
        df_preferencias = dados_brutos["preferencias"].copy()
        df_preferencias['id_docente'] = df_preferencias['id_docente'].str.strip()
        lista_professores = df_professores['id_docente'].tolist()
        lista_disciplinas = df_disciplinas['id_disciplina'].tolist()
        
        # Cria o dicionário de preferências completo
        prof_que_responderam = df_preferencias['id_docente'].unique().tolist()
        df_pivot = df_preferencias.pivot(index='id_docente', columns='id_disciplina', values='preferencia')
        df_completo = df_pivot.reindex(index=lista_professores, columns=lista_disciplinas)
        for prof in lista_professores:
            if prof in prof_que_responderam:
                df_completo.loc[prof] = df_completo.loc[prof].fillna(0)
            else:
                df_completo.loc[prof] = df_completo.loc[prof].fillna(1)
        
        self.dados_preparados = {
            "professores": df_professores.set_index('id_docente'),
            "disciplinas": df_disciplinas.set_index('id_disciplina'),
            "preferencias": df_completo.astype(int).to_dict(orient='index')
        }
        print("Dados de contexto carregados pelo analisador.")

    def avaliar(self, df_alocacao: pd.DataFrame):
        """
        Analisa uma solução de alocação fornecida como um DataFrame.

        Args:
            df_alocacao (pd.DataFrame): DataFrame contendo a alocação a ser avaliada.
                                        Deve conter as colunas 'id_docente' e 'id_disciplina'.

        Returns:
            dict: Um dicionário com as métricas da solução.
        """
        if not all(col in df_alocacao.columns for col in ['id_docente', 'id_disciplina']):
            raise ValueError("O DataFrame de alocação deve conter as colunas 'id_docente' and 'id_disciplina'.")

        df_analise = df_alocacao.copy()
        
        # --- 1. Cálculo das Métricas Base ---
        # Adiciona as informações de preferência e carga horária a cada alocação
        df_analise['preferencia'] = df_analise.apply(
            lambda row: self.dados_preparados['preferencias'][row['id_docente']][row['id_disciplina']],
            axis=1
        )
        df_analise['carga_horaria'] = df_analise['id_disciplina'].map(
            self.dados_preparados['disciplinas']['carga_horaria']
        )

        # --- 2. Escore da Solução ---
        escore_total = df_analise['preferencia'].sum()

        # --- 3. Contagem de Preferências ---
        contagem_prefs = df_analise['preferencia'].value_counts()
        distribuicao_prefs = {
            'pref_3': contagem_prefs.get(3, 0),
            'pref_2': contagem_prefs.get(2, 0),
            'pref_1': contagem_prefs.get(1, 0),
            'pref_0': contagem_prefs.get(0, 0)
        }
        


        return {
            "escore_total": escore_total,
            "distribuicao_preferencias": distribuicao_prefs
            }