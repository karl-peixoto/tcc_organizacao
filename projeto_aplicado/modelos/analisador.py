import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

class AnalisadorDeSolucao:
    """Avalia a qualidade de uma solução de alocação.

    Agora suporta dois modos de inicialização:
    - Carregamento de arquivos na pasta 'dados' (modo original)
    - Injeção direta de DataFrames/estruturas já preparadas (dados_injetados)

    Se dados_injetados for fornecido, a avaliação ocorrerá *exatamente* sobre
    esses dados (útil para cenários perturbados que desejam avaliação consistente).
    """

    def __init__(self, config: dict, dados_injetados: Optional[Dict[str, Any]] = None):
        self.config = config
        self.dados_preparados: Dict[str, Any] = {}
        inj = dados_injetados if dados_injetados is not None else config.get('DADOS_INJETADOS')
        if inj is not None:
            self._preparar_injetados(inj)
        else:
            self._carregar_e_preparar_dados()


    def _normalizar_conflitos(self, df_conf: Optional[pd.DataFrame], df_disc: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df_conf is None:
            return None
        df = df_conf.copy()
        # Restaura índice salvo em CSV (ex: 'Unnamed: 0') se necessário
        if df.index.name is None and df.columns.size > 0:
            try:
                df = df.set_index(df.columns[0])
            except Exception:
                pass
        ids = df_disc['id_disciplina'].astype(str).tolist()
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        df = df.reindex(index=ids, columns=ids).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        # Simetria e diagonal zero
        df.values[np.diag_indices_from(df)] = 0
        df = ((df + df.T) > 0).astype(int)
        return df
    
    # ------------------------ MODO ORIGINAL ------------------------ #
    def _carregar_e_preparar_dados(self):
        try:
            raiz = Path(__file__).parent.parent.parent
            pasta_dados = raiz / "dados"
            dados_brutos = {k: pd.read_csv(pasta_dados / v) for k, v in self.config["ARQUIVOS_DADOS"].items()}
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Erro ao carregar arquivos: {e}")

        df_prof = dados_brutos["professores"].copy()
        df_disc = dados_brutos["disciplinas"].copy()
        df_pref = dados_brutos["preferencias"].copy()
        df_conf = dados_brutos.get("conflitos")

        df_prof['id_docente'] = df_prof['id_docente'].str.strip()
        df_pref['id_docente'] = df_pref['id_docente'].str.strip()

        profs = df_prof['id_docente'].tolist()
        discs = df_disc['id_disciplina'].tolist()

        df_pivot = df_pref.pivot(index='id_docente', columns='id_disciplina', values='preferencia')
        df_comp = df_pivot.reindex(index=profs, columns=discs)
        responded = df_pref['id_docente'].unique().tolist()
        for p in profs:
            fill = 0 if p in responded else 1
            df_comp.loc[p] = df_comp.loc[p].fillna(fill)

        if df_conf is not None:
            df_conf = df_conf.set_index(df_conf.columns[0])

        self.dados_preparados = {
            'professores': df_prof.set_index('id_docente'),
            'disciplinas': df_disc.set_index('id_disciplina'),
            'preferencias': df_comp.astype(int).to_dict(orient='index'),
            'capacidades': df_prof.set_index('id_docente')['max_disciplinas'].to_dict(),
            'matriz_conflitos': df_conf
        }

    # ------------------------ MODO INJETADO ------------------------ #
    def _preparar_injetados(self, dados: Dict[str, Any]):
        """Aceita estruturas já perturbadas/preparadas.

        Espera chaves opcionais:
        - professores: DataFrame com id_docente e max_disciplinas
        - disciplinas: DataFrame com id_disciplina e atributos (inclui carga_horaria)
        - preferencias: pode ser DataFrame pivot (index=prof, columns=disc) ou longa
        - conflitos: DataFrame matriz (index disciplina, colunas disciplina)

        Converte tudo para o formato interno utilizado na avaliação.
        """
        df_prof = dados.get('professores').copy()
        df_disc = dados.get('disciplinas').copy()
        prefs = dados.get('preferencias')
        df_conf = dados.get('conflitos')

        if df_prof is None or df_disc is None or prefs is None:
            raise ValueError("Dados injetados devem incluir 'professores', 'disciplinas' e 'preferencias'.")

        # Professores
        df_prof['id_docente'] = df_prof['id_docente'].astype(str).str.strip()
        df_disc['id_disciplina'] = df_disc['id_disciplina'].astype(str)

        # Preferências: detectar formato
        if isinstance(prefs, pd.DataFrame):
            if {'id_docente', 'id_disciplina', 'preferencia'}.issubset(prefs.columns):
                # formato longo -> pivot
                df_pref_long = prefs.copy()
                df_pref_long['id_docente'] = df_pref_long['id_docente'].astype(str).str.strip()
                df_pivot = df_pref_long.pivot(index='id_docente', columns='id_disciplina', values='preferencia')
            else:
                # assumir pivot já (index=prof, columns=disc)
                df_pivot = prefs.copy()
        else:
            raise ValueError("'preferencias' deve ser DataFrame em formato longo ou pivot.")

        # Completar missing com 0
        profs = df_prof['id_docente'].tolist()
        discs = df_disc['id_disciplina'].tolist()
        df_pivot = df_pivot.reindex(index=profs, columns=discs)
#        df_conf = self._normalizar_conflitos(df_conf, df_disc)

        self.dados_preparados = {
            'professores': df_prof.set_index('id_docente'),
            'disciplinas': df_disc.set_index('id_disciplina'),
            'preferencias': df_pivot.astype(int).to_dict(orient='index'),
            'capacidades': df_prof.set_index('id_docente')['max_disciplinas'].to_dict(),
            'matriz_conflitos': df_conf
        }

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
            raise ValueError("O DataFrame de alocação deve conter as colunas 'id_docente' e 'id_disciplina'.")

        df_analise = df_alocacao.copy()
        
        # --- 1. Cálculo das Métricas Base ---
        # Adiciona as informações de preferência e carga horária a cada alocação
        df_analise['preferencia'] = df_analise.apply(
            lambda r: self.dados_preparados['preferencias'][r['id_docente']][r['id_disciplina']], axis=1
        )
        if 'carga_horaria' in self.dados_preparados['disciplinas'].columns:
            df_analise['carga_horaria'] = df_analise['id_disciplina'].map(
                self.dados_preparados['disciplinas']['carga_horaria']
            )

        # --- 2. Escore da Solução ---
        escore_total = int(df_analise['preferencia'].sum())

        # --- 3. Contagem de Preferências ---
        contagem_prefs = df_analise['preferencia'].value_counts()
        distribuicao_prefs = {
            'pref_3': contagem_prefs.get(3, 0),
            'pref_2': contagem_prefs.get(2, 0),
            'pref_1': contagem_prefs.get(1, 0),
            'pref_0': contagem_prefs.get(0, 0)
        }

        # --- 4. Métricas de restrições de capacidade ---
        capacidades = self.dados_preparados.get('capacidades', {})
        carga_por_prof = df_analise.groupby('id_docente')['id_disciplina'].count().to_dict()
        violacoes_por_prof = {p: max(0, carga_por_prof.get(p, 0) - cap) for p, cap in capacidades.items()}
        total_excesso = int(sum(violacoes_por_prof.values()))
        professores_com_violacao = int(sum(1 for v in violacoes_por_prof.values() if v > 0))

        # --- 5. Contagem de conflitos existentes na solução ---
        matriz_conflitos = self.dados_preparados.get('matriz_conflitos')
        conflitos_total = 0
        if matriz_conflitos is not None and not df_analise.empty:
            # Para cada professor, considerar pares de disciplinas
            for prof, grupo in df_analise.groupby('id_docente'):
                disciplinas_prof = grupo['id_disciplina'].tolist()
                for i in range(len(disciplinas_prof)):
                    di = disciplinas_prof[i]
                    for j in range(i + 1, len(disciplinas_prof)):
                        dj = disciplinas_prof[j]
                        try:
                            if matriz_conflitos.loc[di, dj] == 1:
                                conflitos_total += 1
                        except Exception:
                            continue

        # --- 6. Montagem ---
        metricas_capacidade = {
            'violacoes_capacidade_total_excesso': total_excesso,
            'professores_com_violacao': professores_com_violacao
        }
        metricas_conflitos = {
            'conflitos_totais': conflitos_total
        }
        


        return {
            'escore_total': escore_total,
            'distribuicao_preferencias': distribuicao_prefs,
            **metricas_capacidade,
            **metricas_conflitos
        }