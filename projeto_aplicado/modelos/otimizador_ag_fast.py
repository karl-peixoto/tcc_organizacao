import numpy as np
import pandas as pd
import random
import time
from .otimizador_base import Otimizador

class OtimizadorAGFast(Otimizador):
    """Versão altamente otimizada do Algoritmo Genético.

    Mantém a mesma interface e chaves de retorno da versão original (`OtimizadorAG`).
    Principais otimizações:
      - Representação da população em `numpy.ndarray` (int) ao invés de listas de listas.
      - Cálculo vetorizado do fitness (preferências, cargas e conflitos) para toda a população.
      - Pré-computação de estruturas (matriz de preferências, horas, conflitos, mapping ids->índices).
      - Redução de chamadas a dicionários dentro dos loops principais.
      - Eliminação de `prints` durante execução (silencioso).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        params = self.config.get("AG_PARAMS", {})
        self.n_populacao = params.get("n_populacao", 50)
        self.n_geracoes = params.get("n_geracoes", 200)
        self.taxa_crossover = params.get("taxa_crossover", 0.8)
        self.taxa_mutacao = params.get("taxa_mutacao", 0.02)
        self.tamanho_torneio = params.get("tamanho_torneio", 3)
        self.fator_penalidade = params.get("fator_penalidade", 1000)
        self.tamanho_elite = params.get("tamanho_elite", 2)

        # Mapeamentos e estruturas constantes do problema reduzido
        self.prof_ids = self.dados_preparados["professores"]
        self.disc_ids = self.dados_preparados["disciplinas"]
        self.n_prof = len(self.prof_ids)
        self.n_disc = len(self.disc_ids)
        self.prof_index = {pid: i for i, pid in enumerate(self.prof_ids)}
        self.disc_index = {did: i for i, did in enumerate(self.disc_ids)}

        # Matriz de preferências: shape (n_prof, n_disc)
        self.pref_matrix = np.zeros((self.n_prof, self.n_disc), dtype=np.int16)
        for p in self.prof_ids:
            pi = self.prof_index[p]
            pref_dict = self.dados_preparados["preferencias"][p]
            for d in self.disc_ids:
                di = self.disc_index[d]
                self.pref_matrix[pi, di] = pref_dict[d]

        # Cada disciplina tem custo unitário: 1 (limite por professor medido em número de disciplinas)
        self.ch_disciplina = np.array([self.dados_preparados["ch_disciplinas"][d] for d in self.disc_ids], dtype=np.int16)
        # Limite de disciplinas por professor
        self.ch_max = np.array([self.dados_preparados["ch_max"][p] for p in self.prof_ids], dtype=np.int16)

        # Matriz de conflitos (disciplinas x disciplinas) em numpy para acessos rápidos
        matriz_conflitos_df = self.dados_preparados["matriz_conflitos"]
        self.conflicts = matriz_conflitos_df.values.astype(np.int8)
        # Índices dos pares que possuem conflito (upper triangle) para contagem vetorizada
        iu, ju = np.where(np.triu(self.conflicts, 1) == 1)
        self.conflict_pairs_i = iu
        self.conflict_pairs_j = ju

        # Estado evolucionário
        self.populacao = None  # ndarray shape (n_populacao, n_disc) com índices de professor
        self.melhor_solucao_global = None
        self.melhor_fitness_global = -np.inf
        self.metricas_iteracao = []

    # ------------------------------------------------------------------
    # Construção de população inicial
    # ------------------------------------------------------------------
    def _escolher_professor_valido(self, disciplina_idx: int, solucao_prof_por_disc: np.ndarray, cargas_atuais: np.ndarray) -> int:
        """Replica a lógica original com restrições de carga e conflitos, retorna índice do professor."""
        disc_id = self.disc_ids[disciplina_idx]
        ch_d = self.ch_disciplina[disciplina_idx]

        # Professores que ainda comportam a disciplina
        candidatos = np.where(cargas_atuais + ch_d <= self.ch_max)[0]
        if candidatos.size == 0:
            return random.randrange(self.n_prof)

        # Filtra por conflito de horário (disciplina já alocada para prof conflitando)
        # Para cada disciplina já alocada, se conflita com a atual, o professor daquela disciplina fica inválido
        disciplinas_alocadas_mask = solucao_prof_por_disc >= 0
        if disciplinas_alocadas_mask.any():
            disciplinas_alocadas_idx = np.where(disciplinas_alocadas_mask)[0]
            conflitos_com_atual = self.conflicts[disciplina_idx, disciplinas_alocadas_idx] == 1
            if conflitos_com_atual.any():
                profs_conflitantes = solucao_prof_por_disc[disciplinas_alocadas_idx[conflitos_com_atual]]
                candidatos = np.setdiff1d(candidatos, profs_conflitantes, assume_unique=False)
                if candidatos.size == 0:
                    return random.randrange(self.n_prof)

        # Preferências ponderam o sorteio
        prefs = self.pref_matrix[candidatos, disciplina_idx]
        soma = prefs.sum()
        if soma <= 0:
            return int(np.random.choice(candidatos))
        probs = prefs / soma
        return int(np.random.choice(candidatos, p=probs))

    def _gerar_populacao_inicial(self):
        self.populacao = np.empty((self.n_populacao, self.n_disc), dtype=np.int16)
        for i in range(self.n_populacao):
            sol = np.full(self.n_disc, -1, dtype=np.int16)
            cargas = np.zeros(self.n_prof, dtype=np.int16)
            ordem = np.arange(self.n_disc)
            np.random.shuffle(ordem)
            for d_idx in ordem:
                p_idx = self._escolher_professor_valido(d_idx, sol, cargas)
                sol[d_idx] = p_idx
                cargas[p_idx] += self.ch_disciplina[d_idx]
            self.populacao[i] = sol

    # ------------------------------------------------------------------
    # Cálculo vetorizado de fitness
    # ------------------------------------------------------------------
    def _calcular_fitness_populacao(self, pop: np.ndarray):
        """Retorna tupla (fitness, soma_pref, penal_carga, penal_conflitos).
        Mantém lógica anterior, apenas expõe componentes para métricas.
        """
        scores_pref = self.pref_matrix[pop, np.arange(self.n_disc)]
        soma_pref = scores_pref.sum(axis=1).astype(np.int64)

        excesso = np.zeros(pop.shape[0], dtype=np.int64)
        for i, individuo in enumerate(pop):
            cargas_prof = np.bincount(individuo, weights=self.ch_disciplina, minlength=self.n_prof)
            excesso_ind = cargas_prof - self.ch_max
            excesso[i] = excesso_ind[excesso_ind > 0].sum()
        penal_carga = excesso * self.fator_penalidade

        if self.conflict_pairs_i.size:
            same_prof = (pop[:, self.conflict_pairs_i] == pop[:, self.conflict_pairs_j])
            num_conflitos = same_prof.sum(axis=1)
        else:
            num_conflitos = np.zeros(pop.shape[0], dtype=np.int64)
        penal_conflitos = num_conflitos * self.fator_penalidade

        fitness = soma_pref - penal_carga - penal_conflitos
        return fitness, soma_pref, penal_carga, penal_conflitos

    # ------------------------------------------------------------------
    # Operadores Genéticos
    # ------------------------------------------------------------------
    def _selecao_por_torneio(self, fitness: np.ndarray) -> np.ndarray:
        idxs = np.random.choice(self.n_populacao, size=self.tamanho_torneio, replace=False)
        sub_fit = fitness[idxs]
        vencedor = idxs[np.argmax(sub_fit)]
        return self.populacao[vencedor]

    def _crossover(self, pai1: np.ndarray, pai2: np.ndarray) -> tuple:
        if random.random() > self.taxa_crossover:
            return pai1.copy(), pai2.copy()
        ponto = random.randint(1, self.n_disc - 1)
        f1 = np.concatenate([pai1[:ponto], pai2[ponto:]])
        f2 = np.concatenate([pai2[:ponto], pai1[ponto:]])
        return f1, f2

    def _mutacao(self, crom: np.ndarray) -> np.ndarray:
        """Mutação adaptativa versão vetorizada (Shift vs Swap).

        Para cada gene sorteado:
          - Escolhe professor alvo ponderado pela preferência da disciplina.
          - Tenta Shift (capacidade + sem conflito).
          - Senão tenta Swap com disciplina de menor preferência do alvo movendo-a ao professor de origem.
          - Fallback: escolha válida via método original.
        """
        mut = crom.copy()

        # Construir cargas e lista de disciplinas por professor
        cargas = np.zeros(self.n_prof, dtype=np.int16)
        disciplinas_por_prof = [[] for _ in range(self.n_prof)]
        for d_idx, p_idx in enumerate(mut):
            cargas[p_idx] += self.ch_disciplina[d_idx]
            disciplinas_por_prof[p_idx].append(d_idx)

        def sem_conflito_idx(prof_idx: int, disc_idx_nova: int, disc_indices_atual: list) -> bool:
            for d in disc_indices_atual:
                if d == disc_idx_nova:
                    continue
                if self.conflicts[d, disc_idx_nova] == 1:
                    return False
            return True

        for g in range(self.n_disc):
            if random.random() >= self.taxa_mutacao:
                continue
            disc_idx = g
            prof_origem_idx = mut[g]

            # Sorteio ponderado do alvo
            prefs = self.pref_matrix[:, disc_idx]
            soma = prefs.sum()
            if soma <= 0:
                prof_alvo_idx = random.randrange(self.n_prof)
            else:
                probs = prefs / soma
                prof_alvo_idx = int(np.random.choice(np.arange(self.n_prof), p=probs))

            if prof_alvo_idx == prof_origem_idx:
                continue

            # Tentativa Shift
            capacidade_ok = (cargas[prof_alvo_idx] + self.ch_disciplina[disc_idx]) <= self.ch_max[prof_alvo_idx]
            conflitos_ok = sem_conflito_idx(prof_alvo_idx, disc_idx, disciplinas_por_prof[prof_alvo_idx])
            if capacidade_ok and conflitos_ok:
                # Shift
                disciplinas_por_prof[prof_origem_idx].remove(disc_idx)
                cargas[prof_origem_idx] -= self.ch_disciplina[disc_idx]
                disciplinas_por_prof[prof_alvo_idx].append(disc_idx)
                cargas[prof_alvo_idx] += self.ch_disciplina[disc_idx]
                mut[g] = prof_alvo_idx
                continue

            # Tentativa Swap
            swap_list = disciplinas_por_prof[prof_alvo_idx]
            if swap_list:
                # Ordena por menor preferência do alvo
                swap_candidates = sorted(swap_list, key=lambda d: self.pref_matrix[prof_alvo_idx, d])
                swap_done = False
                for d_swap in swap_candidates:
                    if d_swap == disc_idx:
                        continue
                    # Capacidade origem após troca
                    carga_origem_pos = cargas[prof_origem_idx] - self.ch_disciplina[disc_idx] + self.ch_disciplina[d_swap]
                    if carga_origem_pos > self.ch_max[prof_origem_idx]:
                        continue
                    # Conflitos origem (removendo disc_idx antes de adicionar d_swap)
                    origem_rest = [d for d in disciplinas_por_prof[prof_origem_idx] if d != disc_idx]
                    if not sem_conflito_idx(prof_origem_idx, d_swap, origem_rest):
                        continue
                    # Conflitos alvo (removendo d_swap antes de adicionar disc_idx)
                    alvo_rest = [d for d in swap_list if d != d_swap]
                    if not sem_conflito_idx(prof_alvo_idx, disc_idx, alvo_rest):
                        continue
                    # Executa Swap
                    mut[g] = prof_alvo_idx
                    swap_pos = d_swap
                    mut[swap_pos] = prof_origem_idx
                    disciplinas_por_prof[prof_origem_idx].remove(disc_idx)
                    disciplinas_por_prof[prof_alvo_idx].remove(d_swap)
                    disciplinas_por_prof[prof_origem_idx].append(d_swap)
                    disciplinas_por_prof[prof_alvo_idx].append(disc_idx)
                    cargas[prof_origem_idx] = cargas[prof_origem_idx] - self.ch_disciplina[disc_idx] + self.ch_disciplina[d_swap]
                    cargas[prof_alvo_idx] = cargas[prof_alvo_idx] - self.ch_disciplina[d_swap] + self.ch_disciplina[disc_idx]
                    swap_done = True
                    break
                if swap_done:
                    continue

            # Fallback: reconstruir restrições e usar método original
            sol_parcial = mut.copy()
            sol_parcial[g] = -1
            cargas_fallback = np.zeros(self.n_prof, dtype=np.int16)
            for d_idx2, p_idx2 in enumerate(sol_parcial):
                if p_idx2 >= 0:
                    cargas_fallback[p_idx2] += self.ch_disciplina[d_idx2]
            novo_prof = self._escolher_professor_valido(disc_idx, sol_parcial, cargas_fallback)
            if novo_prof != prof_origem_idx:
                # Atualiza estruturas
                disciplinas_por_prof[prof_origem_idx].remove(disc_idx)
                cargas[prof_origem_idx] -= self.ch_disciplina[disc_idx]
                disciplinas_por_prof[novo_prof].append(disc_idx)
                cargas[novo_prof] += self.ch_disciplina[disc_idx]
                mut[g] = novo_prof
        return mut

    # ------------------------------------------------------------------
    # Formatação da melhor solução
    # ------------------------------------------------------------------
    def _formatar_solucao_final(self):
        if self.melhor_solucao_global is None:
            return None
        alocacoes = []
        for d_idx, p_idx in enumerate(self.melhor_solucao_global):
            alocacoes.append({
                "id_disciplina": self.disc_ids[d_idx],
                "id_docente": self.prof_ids[p_idx],
                "preferencia": int(self.pref_matrix[p_idx, d_idx])
            })
        df_alocacao = pd.DataFrame(alocacoes)
        return {"alocacao_final": df_alocacao, "valor_objetivo": float(self.melhor_fitness_global)}

    # ------------------------------------------------------------------
    # Núcleo de resolução
    # ------------------------------------------------------------------
    def _resolver_nucleo(self, callback_iteracao=None):
        self._gerar_populacao_inicial()
        stagnation_limit = self.config.get("AG_PARAMS", {}).get("stagnation_limit", None)
        stagnation_count = 0
        prev_best = -np.inf
        for g in range(self.n_geracoes):
            t_start = time.perf_counter()
            fitness, soma_pref, penal_carga, penal_conflitos = self._calcular_fitness_populacao(self.populacao)
            melhor_idx = int(np.argmax(fitness))
            melhor_fit = float(fitness[melhor_idx])
            if melhor_fit > self.melhor_fitness_global:
                self.melhor_fitness_global = melhor_fit
                self.melhor_solucao_global = self.populacao[melhor_idx].copy()
            # Controle de estagnação
            if melhor_fit <= prev_best:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best = melhor_fit

            nova_pop = []
            if self.tamanho_elite > 0:
                elite_indices = np.argsort(fitness)[-self.tamanho_elite:][::-1]
                for ei in elite_indices:
                    nova_pop.append(self.populacao[ei].copy())
            while len(nova_pop) < self.n_populacao:
                p1 = self._selecao_por_torneio(fitness)
                p2 = self._selecao_por_torneio(fitness)
                f1, f2 = self._crossover(p1, p2)
                nova_pop.append(self._mutacao(f1))
                if len(nova_pop) < self.n_populacao:
                    nova_pop.append(self._mutacao(f2))
            self.populacao = np.vstack(nova_pop[:self.n_populacao])

            # Estatísticas adicionais
            metrica = {
                "geracao": g + 1,
                "melhor_geracao": melhor_fit,
                "melhor_global": float(self.melhor_fitness_global),
                "media_geracao": float(fitness.mean()),
                "pior_geracao": float(fitness.min()),
                "std_geracao": float(fitness.std()),
                "soma_preferencias_melhor": float(soma_pref[melhor_idx]),
                "penalidade_carga_melhor": float(penal_carga[melhor_idx]),
                "penalidade_conflito_melhor": float(penal_conflitos[melhor_idx]),
                "penalidade_carga_total": float(penal_carga.sum()),
                "penalidade_conflito_total": float(penal_conflitos.sum()),
                "melhoria_relativa": float(melhor_fit - prev_best),
                "tempo_geracao_seg": time.perf_counter() - t_start,
                "stagnation_count": stagnation_count
            }
            self.metricas_iteracao.append(metrica)
            if callback_iteracao:
                try:
                    callback_iteracao(metrica)
                except Exception:
                    pass

            if stagnation_limit is not None and stagnation_count >= stagnation_limit:
                break

        resultado = self._formatar_solucao_final()
        if resultado is None:
            return None
        resultado["metricas_iteracao"] = self.metricas_iteracao
        return resultado
