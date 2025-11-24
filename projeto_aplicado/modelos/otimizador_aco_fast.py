import numpy as np
import pandas as pd
import time
from .otimizador_base import Otimizador

class OtimizadorACOFast(Otimizador):
    """Versão otimizada do ACO.
    Alterações principais:
      - Uso de arrays numpy para feromônio e heurística
      - Probabilidades calculadas sem DataFrames
      - Remoção de prints para execução silenciosa
    Mantém mesmas chaves de saída: alocacao_final, valor_objetivo, metricas_iteracao.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        params = self.config.get("ACO_PARAMS", {})
        self.n_formigas = params.get("n_formigas", 10)
        self.n_geracoes = params.get("n_geracoes", 100)
        self.alfa = params.get("alfa", 1.0)
        self.beta = params.get("beta", 2.0)
        self.taxa_evaporacao = params.get("taxa_evaporacao", 0.1)

        self.prof_ids = self.dados_preparados["professores"]
        self.disc_ids = self.dados_preparados["disciplinas"]
        self.n_prof = len(self.prof_ids)
        self.n_disc = len(self.disc_ids)
        self.prof_index = {pid: i for i, pid in enumerate(self.prof_ids)}
        self.disc_index = {did: i for i, did in enumerate(self.disc_ids)}

        # Feromônio e heurística
        self.feromonio = np.full((self.n_prof, self.n_disc), 1.0, dtype=np.float32)
        self.heuristica = np.zeros((self.n_prof, self.n_disc), dtype=np.float32)
        for p in self.prof_ids:
            pi = self.prof_index[p]
            prefs = self.dados_preparados["preferencias"][p]
            for d in self.disc_ids:
                di = self.disc_index[d]
                self.heuristica[pi, di] = prefs[d] + 0.1

        self.ch_disciplina = np.array([self.dados_preparados["ch_disciplinas"][d] for d in self.disc_ids], dtype=np.int16)
        self.ch_max = np.array([self.dados_preparados["ch_max"][p] for p in self.prof_ids], dtype=np.int16)
        self.conflicts = self.dados_preparados["matriz_conflitos"].values.astype(np.int8)
        iu, ju = np.where(np.triu(self.conflicts, 1) == 1)
        self.conflict_pairs_i = iu
        self.conflict_pairs_j = ju

        self.melhor_solucao_global = None
        self.melhor_qualidade_global = -1
        self.metricas_iteracao = []
        # Parâmetros adicionais para controle de estagnação (opcional)
        self.stagnation_limit = params.get("stagnation_limit", None)

    def _construir_solucao(self):
        sol = np.full(self.n_disc, -1, dtype=np.int16)
        cargas = np.zeros(self.n_prof, dtype=np.int16)
        ordem = np.arange(self.n_disc)
        np.random.shuffle(ordem)
        qualidade = 0
        for d_idx in ordem:
            ch_d = self.ch_disciplina[d_idx]
            candidatos = np.where(cargas + ch_d <= self.ch_max)[0]
            if candidatos.size == 0:
                return None, -1
            # remove candidatos com conflitos
            if sol.max() >= 0:  # já há algo alocado
                conflitos_prev = self.conflicts[d_idx]
                # disciplinas que conflitam e já têm professor
                conflitando_idx = np.where((conflitos_prev == 1) & (sol >= 0))[0]
                if conflitando_idx.size:
                    profs_conflitantes = sol[conflitando_idx]
                    candidatos = np.setdiff1d(candidatos, profs_conflitantes, assume_unique=False)
                    if candidatos.size == 0:
                        return None, -1
            # atratividade
            fer = self.feromonio[candidatos, d_idx]
            heu = self.heuristica[candidatos, d_idx]
            atr = (fer ** self.alfa) * (heu ** self.beta)
            if atr.sum() <= 0:
                p_sel = int(np.random.choice(candidatos))
            else:
                probs = atr / atr.sum()
                p_sel = int(np.random.choice(candidatos, p=probs))
            sol[d_idx] = p_sel
            cargas[p_sel] += ch_d
            qualidade += int(self.heuristica[p_sel, d_idx] - 0.1)  # remove offset ao somar
        return sol, qualidade

    def _evaporar(self):
        self.feromonio *= (1.0 - self.taxa_evaporacao)

    def _depositar(self, solucao: np.ndarray, qualidade: int):
        if solucao is None:
            return
        # depósito proporcional (simples)
        deposito = 1.0
        for d_idx, p_idx in enumerate(solucao):
            self.feromonio[p_idx, d_idx] += deposito

    def _formatar(self):
        if self.melhor_solucao_global is None:
            return None
        aloc = []
        for d_idx, p_idx in enumerate(self.melhor_solucao_global):
            aloc.append({
                "id_disciplina": self.disc_ids[d_idx],
                "id_docente": self.prof_ids[p_idx],
                "preferencia": int(self.heuristica[p_idx, d_idx] - 0.1)
            })
        df = pd.DataFrame(aloc)
        return {"alocacao_final": df, "valor_objetivo": int(self.melhor_qualidade_global)}

    def _resolver_nucleo(self, callback_iteracao=None):
        prev_best = -np.inf
        stagnation_count = 0
        for g in range(self.n_geracoes):
            t_start = time.perf_counter()
            solucoes = []
            for _ in range(self.n_formigas):
                sol, qualidade = self._construir_solucao()
                if sol is not None:
                    solucoes.append((sol, qualidade))
                    if qualidade > self.melhor_qualidade_global:
                        self.melhor_qualidade_global = qualidade
                        self.melhor_solucao_global = sol.copy()

            self._evaporar()
            if self.melhor_solucao_global is not None:
                self._depositar(self.melhor_solucao_global, self.melhor_qualidade_global)

            if solucoes:
                qualidades = [q for _, q in solucoes]
                melhor_geracao = max(qualidades)
                media_geracao = float(np.mean(qualidades))
                pior_geracao = min(qualidades)
                std_geracao = float(np.std(qualidades))
                # Controle de estagnação (baseado no melhor da geração)
                if melhor_geracao <= prev_best:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                    prev_best = melhor_geracao
                melhoria_relativa = float(melhor_geracao - prev_best)
                soma_pref_melhor = float(melhor_geracao)  # qualidade representa soma das preferências
            else:
                melhor_geracao = None
                media_geracao = None
                pior_geracao = None
                std_geracao = None
                melhoria_relativa = None
                soma_pref_melhor = None

            # Penalidades não são usadas na versão fast (restrições respeitadas na construção)
            penalidade_carga_melhor = 0.0
            penalidade_conflito_melhor = 0.0
            penalidade_carga_total = 0.0
            penalidade_conflito_total = 0.0

            metrica = {
                "geracao": g + 1,
                "melhor_geracao": melhor_geracao if melhor_geracao is not None else None,
                "melhor_global": int(self.melhor_qualidade_global),
                "media_geracao": media_geracao,
                "pior_geracao": pior_geracao,
                "std_geracao": std_geracao,
                "soma_preferencias_melhor": soma_pref_melhor,
                "penalidade_carga_melhor": penalidade_carga_melhor,
                "penalidade_conflito_melhor": penalidade_conflito_melhor,
                "penalidade_carga_total": penalidade_carga_total,
                "penalidade_conflito_total": penalidade_conflito_total,
                "melhoria_relativa": melhoria_relativa,
                "tempo_geracao_seg": time.perf_counter() - t_start,
                "stagnation_count": stagnation_count
            }

            self.metricas_iteracao.append(metrica)
            if callback_iteracao:
                try:
                    callback_iteracao(metrica)
                except Exception:
                    pass

            if self.stagnation_limit is not None and stagnation_count >= self.stagnation_limit:
                break

        resultado = self._formatar()
        if resultado is None:
            return None
        resultado["metricas_iteracao"] = self.metricas_iteracao
        return resultado
