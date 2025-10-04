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
        """Método para construir o objeto do modelo PuLP."""
        if self.dados_preparados is None:
            self._preparar_dados()

        dados = self.dados_preparados
        matriz_conflitos = dados["matriz_conflitos"]
        lista_disciplinas = dados["disciplinas"]
        W = self.config.get("PENALIDADE_W", 4)
        
        modelo = pulp.LpProblem("Alocacao_Docentes_PLI", pulp.LpMaximize)
        #Variável de alocação (principal do modelo)
        x = pulp.LpVariable.dicts("Alocacao", (dados["professores"], dados["disciplinas"]), cat='Binary')
        #Variável de folga (conta o nº de alocações de baixa preferência por professor)
        y = pulp.LpVariable.dicts("Penalidade", dados["professores"], lowBound=0, cat='Integer')
        
        soma_preferencias = pulp.lpSum(dados["preferencias"][p][d] * x[p][d] for p in dados["professores"] for d in dados["disciplinas"])
        soma_penalidades = W * pulp.lpSum(y[p] for p in dados["professores"])
        modelo += soma_preferencias - soma_penalidades

        num_disciplinas = len(lista_disciplinas)

        for p in dados["professores"]:
            #Restrição de carga horária máxima
            modelo += pulp.lpSum(dados["ch_disciplinas"][d] * x[p][d] for d in dados["disciplinas"]) <= dados["ch_max"][p], f"Carga_Max_{p}"
            #Restrição que pune as alocações de 0s
            modelo += pulp.lpSum(x[p][d] for d in dados["disciplinas"] if dados["preferencias"][p][d] == 0) <= y[p], f"Soft_Constraint_{p}"
            #Restrição de choque de horário
            for i in range(num_disciplinas):
                for j in range(i + 1, num_disciplinas):
                    d1 = lista_disciplinas[i]
                    d2 = lista_disciplinas[j]
                
                    if matriz_conflitos.loc[d1, d2] == 1:
                        modelo += (x[p][d1] + x[p][d2] <= 1), f"Conflito_{p}_{d1}_{d2}"


        #Restrição de que cada disciplina deve ter exatamente um professor
        for d in lista_disciplinas:
            modelo += pulp.lpSum(x[p][d] for p in dados["professores"]) == 1, f"Cobertura_{d}"


        self.modelo_pli = modelo
        print("Modelo PLI construído com sucesso.")

    def _extrair_solucao(self):
        """Extrai e formata os resultados do modelo resolvido."""
        if self.modelo_pli.status != pulp.LpStatusOptimal:
            print("Nenhuma solução ótima encontrada para extrair.")
            return

        #Resolve e formata as alocações
        dados = self.dados_preparados
        alocacoes = []
        for p in dados["professores"]:
            for d in dados["disciplinas"]:
                if self.modelo_pli.variablesDict()[f"Alocacao_{p}_{d}"].varValue == 1:
                    alocacoes.append({
                        "id_disciplina": d, "id_docente": p,
                        "preferencia": dados["preferencias"][p][d],
                        "ch_disciplina": dados["ch_disciplinas"][d]
                    })
        df_alocacao = pd.DataFrame(alocacoes)
        
        
        self.resultados = {
            "alocacao_final": df_alocacao
        }
        print("Solução extraída e formatada.")


    def _resolver_nucleo(self):
        """
        Método público principal que orquestra todo o processo de otimização.
        """
        self._construir_modelo()
        print("\nIniciando a resolução do modelo PLI...")
        self.modelo_pli.solve()
        print(f"Status da Solução: {pulp.LpStatus[self.modelo_pli.status]}")
        self._extrair_solucao()
        return self.resultados
    
