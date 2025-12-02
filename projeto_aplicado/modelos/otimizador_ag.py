import pandas as pd
import numpy as np
import random
import time
from .otimizador_base import Otimizador

class OtimizadorAG(Otimizador):
    """
    Implementação OTIMIZADA do Algoritmo Genético (AG) usando NumPy.
    Substitui operações de Pandas e strings por vetores de inteiros.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Parâmetros
        params = self.config.get("AG_PARAMS", {})
        self.n_populacao = params.get("n_populacao", 50)
        self.n_geracoes = params.get("n_geracoes", 200)
        self.taxa_crossover = params.get("taxa_crossover", 0.8)
        self.taxa_mutacao = params.get("taxa_mutacao", 0.02)
        self.tamanho_torneio = params.get("tamanho_torneio", 3)
        self.fator_penalidade = params.get("fator_penalidade", 10)
        self.tamanho_elite = params.get("tamanho_elite", 2)    
        
        # Estruturas Numéricas (Arrays)
        self.arr_pref = None      # Matriz (N_Prof x N_Disc)
        self.arr_ch_cost = None   # Custo de cada disciplina (N_Disc)
        self.arr_ch_max = None    # Capacidade máxima de cada prof (N_Prof)
        
        # Lista de pares de disciplinas que conflitam [(idx_d1, idx_d2), ...]
        # Usado para checagem ultra-rápida de conflitos
        self.pares_conflito_idxs = None 
        
        # Mapeamentos
        self.prof_to_idx = {}
        self.idx_to_prof = {}
        self.disc_to_idx = {}
        self.idx_to_disc = {}
        self.num_profs = 0
        self.num_discs = 0

    def _inicializar_estruturas_numpy(self):
        """Converte dados para matrizes Numpy e inteiros."""
        professores = self.dados_preparados["professores"]
        disciplinas = self.dados_preparados["disciplinas"]
        
        self.num_profs = len(professores)
        self.num_discs = len(disciplinas)

        # 1. Mapeamentos
        self.prof_to_idx = {p: i for i, p in enumerate(professores)}
        self.idx_to_prof = {i: p for i, p in enumerate(professores)}
        self.disc_to_idx = {d: i for i, d in enumerate(disciplinas)}
        self.idx_to_disc = {i: d for i, d in enumerate(disciplinas)}

        # 2. Array de Preferências (N_Prof, N_Disc)
        self.arr_pref = np.zeros((self.num_profs, self.num_discs), dtype=np.float32)
        for p_idx in range(self.num_profs):
            p_id = self.idx_to_prof[p_idx]
            for d_idx in range(self.num_discs):
                d_id = self.idx_to_disc[d_idx]
                self.arr_pref[p_idx, d_idx] = self.dados_preparados["preferencias"][p_id][d_id]

        # 3. Cargas
        self.arr_ch_cost = np.array([self.dados_preparados["ch_disciplinas"][d] for d in disciplinas], dtype=np.int32)
        self.arr_ch_max = np.array([self.dados_preparados["ch_max"][p] for p in professores], dtype=np.int32)

        # 4. Conflitos (Lista de Pares)
        # Em vez de matriz N x N, guardamos apenas quem conflita com quem para iterar rápido
        df_conf = self.dados_preparados["matriz_conflitos"]
        pares = []
        # Percorre apenas o triângulo superior para não duplicar
        for i in range(self.num_discs):
            for j in range(i + 1, self.num_discs):
                d1 = disciplinas[i]
                d2 = disciplinas[j]
                if df_conf.loc[d1, d2] == 1:
                    pares.append([i, j])
        
        if pares:
            self.pares_conflito_idxs = np.array(pares, dtype=np.int32)
        else:
            self.pares_conflito_idxs = np.empty((0, 2), dtype=np.int32)

        # Matriz densa de conflitos para consulta O(1) na mutação
        self.matriz_conflito_bool = np.zeros((self.num_discs, self.num_discs), dtype=bool)
        if len(pares) > 0:
            rows, cols = zip(*pares)
            self.matriz_conflito_bool[rows, cols] = True
            self.matriz_conflito_bool[cols, rows] = True


    def _gerar_individuo_valido(self) -> np.ndarray:
        """Gera um cromossomo (array de índices de professores) tentando respeitar restrições."""
        cromossomo = np.zeros(self.num_discs, dtype=np.int32)
        cargas_atuais = np.zeros(self.num_profs, dtype=np.int32)
        
        indices_disciplinas = np.random.permutation(self.num_discs)
        
        for d_idx in indices_disciplinas:
            custo = self.arr_ch_cost[d_idx]
            
            # Filtro Capacidade
            mask_cap = (cargas_atuais + custo) <= self.arr_ch_max
            candidatos = np.where(mask_cap)[0]
            
            if len(candidatos) == 0:
                # Fallback: escolhe qualquer um aleatoriamente se estiver tudo cheio
                prof_escolhido = np.random.randint(0, self.num_profs)
            else:
                # Sorteio ponderado pela preferência
                prefs = self.arr_pref[candidatos, d_idx]
                soma = prefs.sum()
                if soma > 0:
                    probs = prefs / soma
                    prof_escolhido = np.random.choice(candidatos, p=probs)
                else:
                    prof_escolhido = np.random.choice(candidatos)
            
            cromossomo[d_idx] = prof_escolhido
            cargas_atuais[prof_escolhido] += custo
            
        return cromossomo

    def _calcular_fitness_vetorizado(self, cromossomo: np.ndarray) -> float:
        """
        Calcula fitness usando operações vetoriais puras.
        Cromossomo é um array de int onde cromossomo[d_idx] = p_idx
        """
        # 1. Score de Preferências (Fancy Indexing - muito rápido)
        # Pega a preferencia do prof alocado para cada disciplina e soma
        score_pref = np.sum(self.arr_pref[cromossomo, np.arange(self.num_discs)])

        # 2. Penalidade de Carga
        # np.bincount conta quantas vezes cada professor aparece (ponderado pelo custo)
        cargas_finais = np.bincount(cromossomo, weights=self.arr_ch_cost, minlength=self.num_profs)
        excesso = np.maximum(cargas_finais - self.arr_ch_max, 0).sum()
        
        # 3. Penalidade de Conflitos
        # Verifica pares pré-calculados. Se professor do par[0] == professor do par[1], tem conflito
        n_conflitos = 0
        if self.pares_conflito_idxs.shape[0] > 0:
            p1 = cromossomo[self.pares_conflito_idxs[:, 0]]
            p2 = cromossomo[self.pares_conflito_idxs[:, 1]]
            n_conflitos = np.sum(p1 == p2)

        return score_pref - (excesso * self.fator_penalidade) - (n_conflitos * self.fator_penalidade)

    def _mutacao_numpy(self, cromossomo: np.ndarray) -> np.ndarray:
        """Versão otimizada da mutação Shift/Swap usando índices."""
        # Copia para não alterar o original
        crom = cromossomo.copy()
        
        # Calcula cargas atuais uma vez
        cargas = np.bincount(crom, weights=self.arr_ch_cost, minlength=self.num_profs)
        
        # Prepara lista de disciplinas por professor para consulta rápida
        alocacoes = [[] for _ in range(self.num_profs)]
        for d_idx, p_idx in enumerate(crom):
            alocacoes[p_idx].append(d_idx)

        # Itera sobre genes
        for d_idx in range(self.num_discs):
            if np.random.random() >= self.taxa_mutacao:
                continue
                
            prof_origem = crom[d_idx]
            custo_disc = self.arr_ch_cost[d_idx]

            # 1. Escolha de alvo ponderada
            prefs_coluna = self.arr_pref[:, d_idx]
            soma = prefs_coluna.sum()
            if soma > 0:
                prof_alvo = np.random.choice(self.num_profs, p=prefs_coluna/soma)
            else:
                prof_alvo = np.random.randint(0, self.num_profs)

            if prof_alvo == prof_origem:
                continue

            # Função auxiliar rápida de verificação de conflito usando matriz bool
            def tem_conflito(p_idx_teste, d_idx_nova, lista_alocada):
                # Pega todos conflitos da disciplina nova
                # E verifica intersecção com o que o professor já tem
                for d_existente in lista_alocada:
                    if d_existente == d_idx_nova: continue
                    if self.matriz_conflito_bool[d_existente, d_idx_nova]:
                        return True
                return False

            # 2. Tenta SHIFT
            cap_ok = (cargas[prof_alvo] + custo_disc) <= self.arr_ch_max[prof_alvo]
            
            if cap_ok:
                if not tem_conflito(prof_alvo, d_idx, alocacoes[prof_alvo]):
                    # Executa Shift
                    alocacoes[prof_origem].remove(d_idx)
                    alocacoes[prof_alvo].append(d_idx)
                    cargas[prof_origem] -= custo_disc
                    cargas[prof_alvo] += custo_disc
                    crom[d_idx] = prof_alvo
                    continue # Sucesso, vai pro proximo gene

            # 3. Tenta SWAP
            # Procura no alvo uma disciplina para trocar
            candidatos_troca = alocacoes[prof_alvo]
            random.shuffle(candidatos_troca) # Randomiza para não viciar
            
            swap_feito = False
            for d_troca in candidatos_troca:
                # Verifica conflitos cruzados
                # Se eu mover d_idx para alvo (já testado acima parcialmente, mas agora removendo d_troca)
                
                # Alvo recebendo d_idx (sem d_troca)
                alvo_temp_list = [d for d in alocacoes[prof_alvo] if d != d_troca]
                if tem_conflito(prof_alvo, d_idx, alvo_temp_list):
                    continue
                    
                # Origem recebendo d_troca (sem d_idx)
                origem_temp_list = [d for d in alocacoes[prof_origem] if d != d_idx]
                if tem_conflito(prof_origem, d_troca, origem_temp_list):
                    continue
                
                # Executa Swap
                crom[d_idx] = prof_alvo
                crom[d_troca] = prof_origem
                
                # Atualiza estruturas auxiliares
                alocacoes[prof_origem].remove(d_idx)
                alocacoes[prof_alvo].remove(d_troca)
                alocacoes[prof_origem].append(d_troca)
                alocacoes[prof_alvo].append(d_idx)
                
                # Cargas não mudam se custo for igual (simplificação comum, ou recalcula)
                # Como assumimos custo unitário ou parecido, mantemos simples.
                # Se custos forem diferentes:
                custo_troca = self.arr_ch_cost[d_troca]
                cargas[prof_origem] = cargas[prof_origem] - custo_disc + custo_troca
                cargas[prof_alvo] = cargas[prof_alvo] - custo_troca + custo_disc
                
                swap_feito = True
                break
                
        return crom

    def _resolver_nucleo(self, callback_iteracao=None):
        """Execução principal do AG."""
        
        # Reset
        self.metricas_iteracao = []
        self.melhor_solucao_global = None
        self.melhor_fitness_global = -float('inf')
        self.populacao = [] # Agora lista de numpy arrays
        
        # Inicialização
        self._inicializar_estruturas_numpy()
        
        # Gerar População Inicial
        for _ in range(self.n_populacao):
            self.populacao.append(self._gerar_individuo_valido())
            
        inicio = time.time()
        
        for geracao in range(self.n_geracoes):
            # 1. Avaliação em Batch (List comprehension é rápida o suficiente aqui)
            fitness_pop = [self._calcular_fitness_vetorizado(ind) for ind in self.populacao]
            
            # 2. Monitoramento
            max_fit = max(fitness_pop)
            if max_fit > self.melhor_fitness_global:
                self.melhor_fitness_global = max_fit
                idx_best = fitness_pop.index(max_fit)
                self.melhor_solucao_global = self.populacao[idx_best].copy()
                
            # 3. Elitismo
            nova_pop = []
            if self.tamanho_elite > 0:
                indices_ordenados = np.argsort(fitness_pop)[::-1] # Decrescente
                for i in range(self.tamanho_elite):
                    nova_pop.append(self.populacao[indices_ordenados[i]].copy())
            
            # 4. Seleção e Reprodução
            # Pré-converte para array para indexação rápida no torneio
            arr_fitness = np.array(fitness_pop)
            
            while len(nova_pop) < self.n_populacao:
                # Torneio vetorizado
                # Seleciona K competidores aleatoriamente para Pai 1
                comp1 = np.random.randint(0, len(self.populacao), self.tamanho_torneio)
                pai1_idx = comp1[np.argmax(arr_fitness[comp1])]
                
                comp2 = np.random.randint(0, len(self.populacao), self.tamanho_torneio)
                pai2_idx = comp2[np.argmax(arr_fitness[comp2])]
                
                pai1 = self.populacao[pai1_idx]
                pai2 = self.populacao[pai2_idx]
                
                # Crossover
                if np.random.random() <= self.taxa_crossover:
                    ponto = np.random.randint(1, self.num_discs - 1)
                    filho1 = np.concatenate((pai1[:ponto], pai2[ponto:]))
                    filho2 = np.concatenate((pai2[:ponto], pai1[ponto:]))
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()
                
                # Mutação
                nova_pop.append(self._mutacao_numpy(filho1))
                if len(nova_pop) < self.n_populacao:
                    nova_pop.append(self._mutacao_numpy(filho2))
            
            self.populacao = nova_pop
            
            # Métricas
            metrica = {
                "geracao": geracao + 1,
                "melhor_geracao": max_fit,
                "melhor_global": self.melhor_fitness_global,
                "media_geracao": float(np.mean(fitness_pop))
            }
            self.metricas_iteracao.append(metrica)
            
            if callback_iteracao:
                try: callback_iteracao(metrica)
                except: pass
                
        # Formatação Final
        resultado = self._formatar_solucao_final()
        if resultado is None: return None
        
        resultado["metricas_iteracao"] = self.metricas_iteracao
        resultado["tempo_execucao_segundos"] = time.time() - inicio
        resultado["geracoes_executadas"] = len(self.metricas_iteracao)
        
        return resultado

    def _formatar_solucao_final(self):
        """Reconverte índices para strings e formata saída."""
        if self.melhor_solucao_global is None:
            return None
        
        alocacoes = []
        for d_idx, p_idx in enumerate(self.melhor_solucao_global):
            d_id = self.idx_to_disc[d_idx]
            p_id = self.idx_to_prof[p_idx]
            pref = self.dados_preparados["preferencias"][p_id][d_id]
            
            alocacoes.append({
                "id_disciplina": d_id,
                "id_docente": p_id,
                "preferencia": pref
            })
            
        df_alocacao = pd.DataFrame(alocacoes)
        
        return {
            "alocacao_final": df_alocacao,
            "valor_objetivo": self.melhor_fitness_global
        }