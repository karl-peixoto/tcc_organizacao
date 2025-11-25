from .otimizador_base import Otimizador
import pandas as pd
import pulp

class OtimizadorPLI(Otimizador):
    """
    Classe criada para encapsular todo o processo de otimização de alocação de docentes
    utilizando Programação Linear Inteira (PLI).
    """
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.modelo_pli = None
        print("OtimizadorPLI pronto.")

    def _construir_modelo(self):
        """Constrói o modelo PuLP com objetivo direto: preferencia - penalidade_zero."""
        if self.dados_preparados is None:
            self._preparar_dados()

        dados = self.dados_preparados
        matriz_conflitos = dados["matriz_conflitos"]
        lista_disciplinas = dados["disciplinas"]
        W = self.config.get("PENALIDADE_W", 4)

        modelo = pulp.LpProblem("Alocacao_Docentes_PLI", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("Alocacao", (dados["professores"], dados["disciplinas"]), cat='Binary')

        # Objetivo: soma (preferencia - W * indicador(preferencia==0)) * x
        termos_objetivo = []
        for p in dados["professores"]:
            for d in dados["disciplinas"]:
                pref = dados["preferencias"][p][d]
                penal = W if pref == 0 else 0
                termos_objetivo.append((pref - penal) * x[p][d])
        modelo += pulp.lpSum(termos_objetivo)

        # Restrições de quantidade de disciplinas e conflitos
        num_disciplinas = len(lista_disciplinas)
        for p in dados["professores"]:
            modelo += pulp.lpSum(dados["ch_disciplinas"][d] * x[p][d] for d in dados["disciplinas"]) <= dados["ch_max"][p], f"Limite_Disciplinas_{p}"
            for i in range(num_disciplinas):
                for j in range(i + 1, num_disciplinas):
                    d1 = lista_disciplinas[i]
                    d2 = lista_disciplinas[j]
                    if matriz_conflitos.loc[d1, d2] == 1:
                        modelo += (x[p][d1] + x[p][d2] <= 1), f"Conflito_{p}_{d1}_{d2}"

        for d in lista_disciplinas:
            modelo += pulp.lpSum(x[p][d] for p in dados["professores"]) == 1, f"Cobertura_{d}"

        self.modelo_pli = modelo
        print("Modelo PLI construído com sucesso.")

    def _extrair_solucao(self):
        """Extrai e formata os resultados do modelo resolvido."""
        if self.modelo_pli.status != pulp.LpStatusOptimal:
            print("Nenhuma solução ótima encontrada para extrair.")
            return

        dados = self.dados_preparados
        alocacoes = []
        for p in dados["professores"]:
            for d in dados["disciplinas"]:
                var = self.modelo_pli.variablesDict().get(f"Alocacao_{p}_{d}")
                if var is not None and var.varValue == 1:
                    alocacoes.append({
                        "id_disciplina": d,
                        "id_docente": p,
                        "preferencia": dados["preferencias"][p][d],
                        "ch_disciplina": dados["ch_disciplinas"][d]
                    })
        df_alocacao = pd.DataFrame(alocacoes)

        valor_objetivo = None
        soma_pref = None
        num_zero = None
        try:
            valor_objetivo = pulp.value(self.modelo_pli.objective)
            # Recalcula métricas auxiliares
            soma_pref = sum(dados["preferencias"][p][d] for p in dados["professores"] for d in dados["disciplinas"] if self.modelo_pli.variablesDict().get(f"Alocacao_{p}_{d}") and self.modelo_pli.variablesDict()[f"Alocacao_{p}_{d}"].varValue == 1)
            num_zero = sum(1 for p in dados["professores"] for d in dados["disciplinas"] if dados["preferencias"][p][d] == 0 and self.modelo_pli.variablesDict().get(f"Alocacao_{p}_{d}") and self.modelo_pli.variablesDict()[f"Alocacao_{p}_{d}"].varValue == 1)
        except Exception as e:
            print(f"Não foi possível extrair métricas objetivo: {e}")

        self.resultados = {
            "alocacao_final": df_alocacao,
            "valor_objetivo": valor_objetivo,
            "soma_preferencias": soma_pref,
            "num_alocacoes_preferencia_zero": num_zero,
            "penalidade_total": (num_zero * self.config.get("PENALIDADE_W", 4)) if num_zero is not None else None,
            "metricas_iteracao": []
        }
        print("Solução extraída e formatada.")


    def _resolver_nucleo(self, callback_iteracao=None):
        """Resolve PLI (single step); chama callback final se fornecido."""
        self._construir_modelo()
        print("\nIniciando a resolução do modelo PLI...")
        self.modelo_pli.solve()
        print(f"Status da Solução: {pulp.LpStatus[self.modelo_pli.status]}")
        self._extrair_solucao()
        if callback_iteracao is not None:
            try:
                callback_iteracao({"geracao": 1, "melhor_global": self.resultados.get("valor_objetivo")})
            except Exception as e:
                print(f"Callback PLI falhou: {e}")
        return self.resultados
    
