import pandas as pd
import numpy as np
import time
from .otimizador_base import Otimizador

class OtimizadorACO(Otimizador):
    """
    Implementação OTIMIZADA do ACO usando NumPy para vetorização massiva.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        
        params = self.config.get("ACO_PARAMS", {})
        self.n_formigas = params.get("n_formigas", 10)
        self.n_geracoes = params.get("n_geracoes", 100)
        self.alfa = params.get("alfa", 1.0)
        self.beta = params.get("beta", 2.0)
        self.taxa_evaporacao = params.get("taxa_evaporacao", 0.1)
        self.limite_tempo_segundos = params.get("limite_tempo_segundos", 180)

        # Variáveis de estado
        self.metricas_iteracao = []
        self.melhor_qualidade_global = -1
        self.melhor_solucao_global = None

        # Estruturas Numéricas (Arrays)
        self.arr_feromonio = None
        self.arr_heuristica = None
        self.arr_conflitos = None
        self.arr_ch_prof = None 
        self.arr_ch_disc = None
        
        # Mapeamentos (ID <-> Índice Inteiro)
        self.prof_to_idx = {}
        self.idx_to_prof = {}
        self.disc_to_idx = {}
        self.idx_to_disc = {}
        self.num_profs = 0
        self.num_discs = 0

    def _inicializar_parametros(self):
        """
        Converte DataFrames em Matrizes NumPy e cria mapas de índices
        para acesso O(1) durante a simulação.
        """
        professores = self.dados_preparados["professores"]
        disciplinas = self.dados_preparados["disciplinas"]
        
        self.num_profs = len(professores)
        self.num_discs = len(disciplinas)

        # 1. Mapeamentos Bidirecionais
        self.prof_to_idx = {p: i for i, p in enumerate(professores)}
        self.idx_to_prof = {i: p for i, p in enumerate(professores)}
        self.disc_to_idx = {d: i for i, d in enumerate(disciplinas)}
        self.idx_to_disc = {i: d for i, d in enumerate(disciplinas)}

        # 2. Matrizes Principais (NumPy)
        # Feromônio: (N_Profs x N_Discs) inicializado em 1.0
        self.arr_feromonio = np.ones((self.num_profs, self.num_discs), dtype=np.float64)

        # Heurística: (N_Profs x N_Discs) baseada em preferências + 0.1
        # CORREÇÃO: Preenchimento manual garantindo a orientação correta (Prof x Disc)
        # sem depender da interpretação automática do Pandas que causou o erro.
        raw_heuristica = np.zeros((self.num_profs, self.num_discs))
        pref_dict = self.dados_preparados["preferencias"]
        
        for p_idx in range(self.num_profs):
            p_id = self.idx_to_prof[p_idx]
            for d_idx in range(self.num_discs):
                d_id = self.idx_to_disc[d_idx]
                # Acessa o dicionário original {prof: {disc: valor}}
                raw_heuristica[p_idx, d_idx] = pref_dict[p_id][d_id]
        
        self.arr_heuristica = raw_heuristica + 0.1

        # 3. Restrições e Conflitos
        self.arr_ch_prof = np.array([self.dados_preparados["ch_max"][self.idx_to_prof[i]] for i in range(self.num_profs)])
        self.arr_ch_disc = np.array([self.dados_preparados["ch_disciplinas"][self.idx_to_disc[i]] for i in range(self.num_discs)])

        # Matriz de Conflitos (N_Disc x N_Disc) binária
        df_conf = self.dados_preparados["matriz_conflitos"]
        # .loc garante que a ordem das linhas/colunas do numpy bata com nosso mapeamento idx_to_disc
        self.arr_conflitos = df_conf.loc[disciplinas, disciplinas].values 
        
    def _construir_solucao_formiga_numpy(self) -> tuple:
        """
        Versão vetorizada da construção de solução.
        Usa inteiros e arrays numpy ao invés de strings e dicts.
        """
        # Estado Inicial
        cargas_atuais = np.zeros(self.num_profs, dtype=np.int32)
        solucao_indices = [] 
        qualidade_solucao = 0.0
        
        # Rastreio de alocação (-1 = ninguem)
        alocacao_por_disciplina = np.full(self.num_discs, -1, dtype=np.int32)

        # Ordem aleatória de visitação
        ordem_disciplinas = np.random.permutation(self.num_discs)

        for disc_idx in ordem_disciplinas:
            custo_disc = self.arr_ch_disc[disc_idx]
            
            # --- Passo 1: Filtragem de Candidatos ---
            
            # A. Filtro de Capacidade
            mask_capacidade = (cargas_atuais + custo_disc) <= self.arr_ch_prof
            
            # B. Filtro de Conflitos
            conflitos_da_disc = self.arr_conflitos[disc_idx] # Linha da matriz de conflitos
            idxs_conflitantes = np.where(conflitos_da_disc == 1)[0]
            
            profs_bloqueados = alocacao_por_disciplina[idxs_conflitantes]
            profs_bloqueados = profs_bloqueados[profs_bloqueados != -1]
            
            mask_valido = mask_capacidade.copy()
            if len(profs_bloqueados) > 0:
                mask_valido[profs_bloqueados] = False
            
            candidatos_indices = np.where(mask_valido)[0]

            if len(candidatos_indices) == 0:
                return None, -1 # Solução inviável

            # --- Passo 2: Probabilidades ---
            feromonio_cand = self.arr_feromonio[candidatos_indices, disc_idx]
            heuristica_cand = self.arr_heuristica[candidatos_indices, disc_idx]
            
            atratividade = (feromonio_cand ** self.alfa) * (heuristica_cand ** self.beta)
            
            soma = atratividade.sum()
            if soma == 0:
                idx_escolhido_relativo = np.random.randint(0, len(candidatos_indices))
            else:
                probs = atratividade / soma
                idx_escolhido_relativo = np.random.choice(len(candidatos_indices), p=probs)
            
            prof_idx = candidatos_indices[idx_escolhido_relativo]

            # --- Passo 3: Atualização Local ---
            solucao_indices.append((prof_idx, disc_idx))
            cargas_atuais[prof_idx] += custo_disc
            alocacao_por_disciplina[disc_idx] = prof_idx
            
            # Subtrai 0.1 para recuperar a preferência original limpa
            qualidade_solucao += (self.arr_heuristica[prof_idx, disc_idx] - 0.1)

        return solucao_indices, qualidade_solucao

    def _atualizar_feromonio_numpy(self, solucoes_geracao):
        """Atualização otimizada da matriz de feromônios."""
        if not solucoes_geracao:
            return

        # 1. Evaporação Global
        self.arr_feromonio *= (1.0 - self.taxa_evaporacao)

        # 2. Encontra melhor da geração
        melhor_solucao_tupla = max(solucoes_geracao, key=lambda x: x[1])
        melhor_caminho_indices, melhor_qualidade = melhor_solucao_tupla

        if melhor_caminho_indices is None:
            return

        # 3. Depósito
        deposito = 1.0 / (1.0 + (self.melhor_qualidade_global - melhor_qualidade)) if self.melhor_qualidade_global > 0 else 1.0
        
        # Indexação Fancy do Numpy para atualizar apenas os pares (prof, disc) escolhidos
        profs_idxs, discs_idxs = zip(*melhor_caminho_indices)
        self.arr_feromonio[list(profs_idxs), list(discs_idxs)] += deposito

    def _formatar_solucao_final(self):
        """Reverte os índices inteiros para IDs originais e formata a saída."""
        if self.melhor_solucao_global is None:
            return None
        
        alocacoes = []
        for prof_idx, disc_idx in self.melhor_solucao_global:
            prof_id = self.idx_to_prof[prof_idx]
            disc_id = self.idx_to_disc[disc_idx]
            
            # Busca preferência original no dicionário (seguro)
            pref_original = self.dados_preparados["preferencias"][prof_id][disc_id]
            
            alocacoes.append({
                "id_disciplina": disc_id,
                "id_docente": prof_id,
                "preferencia": pref_original
            })
            
        df_alocacao = pd.DataFrame(alocacoes)
        
        return {
            "alocacao_final": df_alocacao,
            "valor_objetivo": self.melhor_qualidade_global
        }

    def _resolver_nucleo(self, callback_iteracao=None):
        """Método principal de controle (loop de gerações)."""
        
        self.metricas_iteracao = []
        self.melhor_qualidade_global = -1
        self.melhor_solucao_global = None
        
        self._inicializar_parametros()
        
        inicio_execucao = time.time()
        
        for geracao in range(self.n_geracoes):
            solucoes_da_geracao = []
            
            # Loop das Formigas
            for _ in range(self.n_formigas):
                solucao, qualidade = self._construir_solucao_formiga_numpy()
                
                if solucao is not None:
                    solucoes_da_geracao.append((solucao, qualidade))
                    
                    if qualidade > self.melhor_qualidade_global:
                        self.melhor_qualidade_global = qualidade
                        self.melhor_solucao_global = solucao 
            
            self._atualizar_feromonio_numpy(solucoes_da_geracao)
            
            # Coleta de Métricas
            melhor_geracao = None
            media_geracao = None
            
            if solucoes_da_geracao:
                qualidades = [q for _, q in solucoes_da_geracao]
                melhor_geracao = max(qualidades)
                media_geracao = float(np.mean(qualidades))
            
            metrica = {
                "geracao": geracao + 1,
                "melhor_geracao": melhor_geracao,
                "melhor_global": self.melhor_qualidade_global,
                "media_geracao": media_geracao
            }
            self.metricas_iteracao.append(metrica)
            
            if callback_iteracao:
                try:
                    callback_iteracao(metrica)
                except:
                    pass
            
            if (time.time() - inicio_execucao) >= self.limite_tempo_segundos:
                break
                
        resultado = self._formatar_solucao_final()
        if resultado is None:
            return None
            
        resultado["metricas_iteracao"] = self.metricas_iteracao
        resultado["tempo_execucao_segundos"] = time.time() - inicio_execucao
        resultado["geracoes_executadas"] = len(self.metricas_iteracao)
        
        return resultado