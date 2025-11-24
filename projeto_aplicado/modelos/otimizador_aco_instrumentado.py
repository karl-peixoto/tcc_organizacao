import pandas as pd
import numpy as np
from .otimizador_aco import OtimizadorACO

class OtimizadorACOInstrumentado(OtimizadorACO):
    """Versão instrumentada do ACO que registra snapshots completos da matriz de
    feromônio em todas as gerações e dados detalhados da construção das soluções.

    Saídas adicionais em resolver():
      - pheromone_snapshots: lista de DataFrames (index=id_docente, columns=id_disciplina)
        incluindo snapshot inicial (geração 0) e após cada geração (tamanho n_geracoes+1)
      - pheromone_long_df: DataFrame em formato longo com colunas
          ['geracao','id_docente','id_disciplina','feromonio'] para fácil uso em Plotly/Matplotlib.
      - historico_formigas: lista de listas; cada posição i contém a lista de dicts
          das formigas da geração i+1: {'geracao', 'indice_formiga', 'qualidade', 'solucao'}.
      - eventos_melhor_global: lista de dicts com evolução do melhor global.

    Observação: Mantém compatibilidade com chaves da versão base (alocacao_final, valor_objetivo, metricas_iteracao).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.pheromone_snapshots = []          # Lista de DataFrames
        self._pheromone_long_rows = []         # Linhas acumuladas para o DataFrame longo
        self.historico_formigas = []           # Lista por geração com dados das formigas
        self.eventos_melhor_global = []        # Eventos de atualização do melhor global

    def _snapshot_feromonio(self, geracao: int):
        """Captura snapshot completo da matriz de feromônio e armazena também em formato longo."""
        snap = self.matriz_feromonio.copy()
        self.pheromone_snapshots.append(snap)
        # Formato longo
        long_df = snap.stack().rename('feromonio').reset_index()
        long_df.columns = ['id_docente', 'id_disciplina', 'feromonio']
        long_df.insert(0, 'geracao', geracao)
        self._pheromone_long_rows.append(long_df)

    def _registrar_evento_melhor_global(self, geracao: int, qualidade: int):
        self.eventos_melhor_global.append({
            'geracao': geracao,
            'melhor_global': qualidade
        })

    def _construir_solucao_formiga_instrumentada(self) -> dict:
        """Constrói solução de uma formiga retornando dict detalhado.
        Retorna dict com chaves: 'solucao', 'qualidade'."""
        solucao, qualidade = self._construir_solucao_formiga()
        return {
            'solucao': solucao,
            'qualidade': qualidade
        }

    def _resolver_nucleo(self, callback_iteracao=None):  # type: ignore[override]
        # Inicializa estruturas e snapshot inicial (geração 0)
        self._inicializar_parametros()
        self._snapshot_feromonio(geracao=0)

        for geracao in range(1, self.n_geracoes + 1):
            dados_formigas = []
            solucoes_validas = []  # (solucao, qualidade) para atualização de feromônio

            for idx_formiga in range(self.n_formigas):
                info = self._construir_solucao_formiga_instrumentada()
                dados_formigas.append({
                    'geracao': geracao,
                    'indice_formiga': idx_formiga,
                    'qualidade': info['qualidade'],
                    'solucao': info['solucao']
                })
                solucao = info['solucao']
                qualidade = info['qualidade']
                if solucao:
                    solucoes_validas.append((solucao, qualidade))
                    if qualidade > self.melhor_qualidade_global:
                        self.melhor_qualidade_global = qualidade
                        self.melhor_solucao_global = solucao
                        self._registrar_evento_melhor_global(geracao, qualidade)

            self.historico_formigas.append(dados_formigas)
            self._atualizar_feromonio(solucoes_validas)

            # Métricas de geração
            if solucoes_validas:
                qualidades = [q for _, q in solucoes_validas]
                melhor_geracao = max(qualidades)
                media_geracao = float(np.mean(qualidades))
                pior_geracao = min(qualidades)
                std_geracao = float(np.std(qualidades))
                diversidade = len({tuple(sol) for sol, _ in solucoes_validas})
            else:
                melhor_geracao = None
                media_geracao = None
                pior_geracao = None
                std_geracao = None
                diversidade = 0

            metrica = {
                'geracao': geracao,
                'melhor_geracao': melhor_geracao,
                'melhor_global': self.melhor_qualidade_global,
                'media_geracao': media_geracao,
                'pior_geracao': pior_geracao,
                'std_geracao': std_geracao,
                'diversidade_solucoes': diversidade
            }
            self.metricas_iteracao.append(metrica)
            if callback_iteracao is not None:
                try:
                    callback_iteracao(metrica)
                except Exception as e:
                    print(f"Callback geração {geracao} falhou: {e}")

            # Snapshot pós atualização de feromônio desta geração
            self._snapshot_feromonio(geracao=geracao)

        resultado_base = self._formatar_solucao_final()
        if resultado_base is None:
            return None
        resultado_base['metricas_iteracao'] = self.metricas_iteracao
        resultado_base['pheromone_snapshots'] = self.pheromone_snapshots
        resultado_base['pheromone_long_df'] = pd.concat(self._pheromone_long_rows, ignore_index=True)
        resultado_base['historico_formigas'] = self.historico_formigas
        resultado_base['eventos_melhor_global'] = self.eventos_melhor_global
        return resultado_base

    # Métodos utilitários para acesso externo
    def get_pheromone_long(self) -> pd.DataFrame:
        return pd.concat(self._pheromone_long_rows, ignore_index=True)

    def get_pheromone_generation(self, geracao: int) -> pd.DataFrame:
        """Retorna snapshot da geração solicitada (0 = inicial)."""
        if geracao < 0 or geracao >= len(self.pheromone_snapshots):
            raise IndexError('Geração fora do intervalo de snapshots.')
        return self.pheromone_snapshots[geracao].copy()

    def get_formigas_geracao(self, geracao: int) -> list:
        """Retorna lista de dicts com dados das formigas da geração (1-index)."""
        if geracao < 1 or geracao > len(self.historico_formigas):
            raise IndexError('Geração fora do intervalo de histórico de formigas.')
        return self.historico_formigas[geracao - 1]
