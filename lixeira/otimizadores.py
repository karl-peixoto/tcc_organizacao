import pandas as pd
import pulp

class OtimizadorPLI:
    """
    Uma classe para encapsular todo o processo de otimização de alocação de docentes
    utilizando Programação Linear Inteira (PLI).
    """
    def __init__(self, config: dict):
        """
        Inicializa o otimizador com as configurações necessárias.

        Args:
            config (dict): Dicionário contendo os caminhos dos arquivos e parâmetros do modelo.
        """
        self.config = config
        self.dados_brutos = None
        self.dados_preparados = None
        self.modelo_pli = None
        self.resultados = None
        print("OtimizadorPLI instanciado.")

    def _carregar_dados(self):
        """Método 'privado' para carregar os dados dos arquivos CSV."""
        try:
            self.dados_brutos = {nome: pd.read_csv(caminho) for nome, caminho in self.config["ARQUIVOS_DADOS"].items()}
            print("Dados carregados com sucesso.")
        except FileNotFoundError as e:
            print(f"Erro: Arquivo não encontrado - {e}.")
            raise

    def _preparar_dados(self):
        """Método 'privado' para preparar os dados para a modelagem."""
        if self.dados_brutos is None:
            self._carregar_dados()
        
        # Extrai listas de IDs
        lista_professores = self.dados_brutos["professores"]['id_professor'].tolist()
        lista_disciplinas = self.dados_brutos["disciplinas"]['id_disciplina'].tolist()

        # Cria dicionários para acesso rápido aos parâmetros
        ch_max = self.dados_brutos["professores"].set_index('id_professor')['carga_maxima'].to_dict()
        ch_disc = self.dados_brutos["disciplinas"].set_index('id_disciplina')['carga_horaria'].to_dict()
        
        # Transforma o DataFrame de preferências em um dicionário aninhado
        prefs_dict = self.dados_brutos["preferencias"].pivot(
            index='id_professor', columns='id_disciplina', values='preferencia'
        ).to_dict(orient='index')

        self.dados_preparados = {
            "professores": lista_professores, "disciplinas": lista_disciplinas,
            "ch_max": ch_max, "ch_disciplinas": ch_disc, "preferencias": prefs_dict,
        }
        print("Dados preparados para a modelagem.")

    def _construir_modelo(self):
        """Método 'privado' para construir o objeto do modelo PuLP."""
        if self.dados_preparados is None:
            self._preparar_dados()

        dados = self.dados_preparados
        W = self.config["PENALIDADE_W"]
        
        modelo = pulp.LpProblem("Alocacao_Docentes_PLI", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("Alocacao", (dados["professores"], dados["disciplinas"]), cat='Binary')
        y = pulp.LpVariable.dicts("Penalidade", dados["professores"], lowBound=0, cat='Integer')
        
        soma_preferencias = pulp.lpSum(dados["preferencias"][p][d] * x[p][d] for p in dados["professores"] for d in dados["disciplinas"])
        soma_penalidades = W * pulp.lpSum(y[p] for p in dados["professores"])
        modelo += soma_preferencias - soma_penalidades

        for d in dados["disciplinas"]:
            modelo += pulp.lpSum(x[p][d] for p in dados["professores"]) == 1, f"Cobertura_{d}"

        for p in dados["professores"]:
            modelo += pulp.lpSum(dados["ch_disciplinas"][d] * x[p][d] for d in dados["disciplinas"]) <= dados["ch_max"][p], f"Carga_Max_{p}"
            modelo += pulp.lpSum(x[p][d] for d in dados["disciplinas"] if dados["preferencias"][p][d] == 1) <= y[p], f"Soft_Constraint_{p}"

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
                if self.modelo_pli.variablesDict()[f"Alocacao_{p}_{d}"].varValue == 1:
                    alocacoes.append({
                        "Disciplina": d, "Professor Alocado": p,
                        "Preferencia Atendida": dados["preferencias"][p][d],
                        "Carga Horaria Disciplina": dados["ch_disciplinas"][d]
                    })
        df_alocacao = pd.DataFrame(alocacoes)
        
        df_carga = df_alocacao.groupby("Professor Alocado")["Carga Horaria Disciplina"].sum().reset_index()
        df_carga = df_carga.rename(columns={"Carga Horaria Disciplina": "Carga Horaria Atribuida"})
        df_prof = self.dados_brutos["professores"].rename(columns={'id_professor': 'Professor Alocado'})
        df_carga = df_carga.merge(df_prof[['Professor Alocado', 'carga_maxima']], on="Professor Alocado", how="right").fillna(0)
        df_carga.rename(columns={'carga_maxima': 'Carga Horaria Maxima'}, inplace=True)
        
        self.resultados = {
            "alocacao_final": df_alocacao, "carga_horaria": df_carga,
            "valor_objetivo": pulp.value(self.modelo_pli.objective)
        }
        print("Solução extraída e formatada.")


    def resolver(self):
        """
        Método público principal que orquestra todo o processo de otimização.
        """
        self._construir_modelo()
        print("\nIniciando a resolução do modelo PLI...")
        self.modelo_pli.solve()
        print(f"Status da Solução: {pulp.LpStatus[self.modelo_pli.status]}")
        self._extrair_solucao()
        return self.resultados

# --- Exemplo de como usar a classe ---
if __name__ == "__main__":
    CONFIG = {
        "ARQUIVOS_DADOS": {
            "disciplinas": "disciplinas_simuladas.csv",
            "professores": "professores_simulados.csv",
            "preferencias": "preferencias_simuladas.csv",
        },
        "PENALIDADE_W": 4.0,
    }

    # 1. Instanciar o otimizador
    otimizador = OtimizadorPLI(config=CONFIG)

    # 2. Resolver o problema (a classe cuida de todos os passos internos)
    resultados_finais = otimizador.resolver()

    # 3. Analisar os resultados
    if resultados_finais:
        print("\n--- RESULTADO FINAL ---")
        print(resultados_finais["alocacao_final"])