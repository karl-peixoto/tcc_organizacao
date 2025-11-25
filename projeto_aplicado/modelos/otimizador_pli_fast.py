import pandas as pd
import pulp
from .otimizador_base import Otimizador

class OtimizadorPLIFast(Otimizador):
    """Versão otimizada da PLI.
    Melhorias:
      - Construção de listas por compreensão ao invés de loops aninhados pesados.
      - Remoção de prints.
      - Solver chamado com `msg=False` para execução silenciosa.
      - Mantém chaves de saída originais.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.modelo = None

    def _construir(self):
        dados = self.dados_preparados
        W = self.config.get("PENALIDADE_W", 4)
        modelo = pulp.LpProblem("Alocacao_Docentes_PLI_FAST", pulp.LpMaximize)
        profs = dados["professores"]
        discs = dados["disciplinas"]
        x = pulp.LpVariable.dicts("A", (profs, discs), cat="Binary")

        # Objetivo vetorizado
        termos = []
        preferencias = dados["preferencias"]
        for p in profs:
            pref_p = preferencias[p]
            for d in discs:
                pref = pref_p[d]
                penal = W if pref == 0 else 0
                termos.append((pref - penal) * x[p][d])
        modelo += pulp.lpSum(termos)

        # Restrição de quantidade de disciplinas por professor (cada disciplina = 1)
        ch_disc = dados["ch_disciplinas"]
        for p in profs:
            modelo += pulp.lpSum(ch_disc[d] * x[p][d] for d in discs) <= dados["ch_max"][p]

        # Conflitos (pares) - iterar somente conflitos reais
        matriz_conflitos = dados["matriz_conflitos"]
        disc_list = discs
        n_disc = len(disc_list)
        for i in range(n_disc):
            di = disc_list[i]
            linha = matriz_conflitos.loc[di]
            # Só percorre j>i e células ==1
            confl_indices = [j for j in range(i+1, n_disc) if linha[disc_list[j]] == 1]
            if confl_indices:
                for p in profs:
                    for j in confl_indices:
                        dj = disc_list[j]
                        modelo += x[p][di] + x[p][dj] <= 1

        # Cobertura: cada disciplina deve ter exatamente um professor
        for d in discs:
            modelo += pulp.lpSum(x[p][d] for p in profs) == 1

        self.modelo = modelo
        self.x = x

    def _extrair(self):
        if self.modelo.status != pulp.LpStatusOptimal:
            return None
        dados = self.dados_preparados
        profs = dados["professores"]
        discs = dados["disciplinas"]
        preferencias = dados["preferencias"]
        ch_disc = dados["ch_disciplinas"]
        aloc = []
        var_dict = self.modelo.variablesDict()
        for p in profs:
            pref_p = preferencias[p]
            for d in discs:
                var = var_dict.get(f"A_{p}_{d}")
                if var and var.varValue == 1:
                    aloc.append({
                        "id_disciplina": d,
                        "id_docente": p,
                        "preferencia": pref_p[d],
                        "ch_disciplina": ch_disc[d]
                    })
        df = pd.DataFrame(aloc)
        valor_obj = pulp.value(self.modelo.objective)
        num_zero = sum(1 for a in aloc if a["preferencia"] == 0)
        soma_pref = sum(a["preferencia"] for a in aloc)
        return {
            "alocacao_final": df,
            "valor_objetivo": valor_obj,
            "soma_preferencias": soma_pref,
            "num_alocacoes_preferencia_zero": num_zero,
            "penalidade_total": num_zero * self.config.get("PENALIDADE_W", 4),
            "metricas_iteracao": []
        }

    def _resolver_nucleo(self, callback_iteracao=None):
        self._construir()
        self.modelo.solve(pulp.PULP_CBC_CMD(msg=False))
        resultados = self._extrair()
        return resultados
