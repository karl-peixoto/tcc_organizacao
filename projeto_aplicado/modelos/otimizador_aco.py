import pandas as pd
import numpy as np
from .otimizador_base import Otimizador

class OtimizadorACO(Otimizador):
    """
    Implementação do otimizador de alocação de docentes usando a meta-heurística
    de Otimização por Colônia de Formigas (ACO).
    
    Herda a lógica de carregamento e preparação de dados da classe Otimizador.
    """
    def __init__(self, config: dict):
        """
        Inicializa o otimizador ACO.

        Args:
            config (dict): Dicionário de configuração contendo:
                - ARQUIVOS_DADOS: Caminhos para os CSVs.
                - ACO_PARAMS: Dicionário com os hiperparâmetros do ACO:
                    - n_formigas (int): Número de formigas por geração.
                    - n_geracoes (int): Número de gerações (iterações).
                    - alfa (float): Importância do feromônio.
                    - beta (float): Importância da informação heurística.
                    - taxa_evaporacao (float): Taxa de evaporação do feromônio (entre 0 e 1).
        """
        super().__init__(config)
        
        params = self.config.get("ACO_PARAMS", {})
        self.n_formigas = params.get("n_formigas", 10)
        self.n_geracoes = params.get("n_geracoes", 100)
        self.alfa = params.get("alfa", 1.0)
        self.beta = params.get("beta", 2.0)
        self.taxa_evaporacao = params.get("taxa_evaporacao", 0.1)

        
        self.matriz_feromonio = None
        self.info_heuristica = None
        self.melhor_solucao_global = None
        self.melhor_qualidade_global = -1

        print("OtimizadorACO pronto.")

    def _inicializar_parametros(self):
        """
        Prepara as matrizes de feromônio e informação heurística antes do início da otimização.
        """
        professores = self.dados_preparados["professores"]
        disciplinas = self.dados_preparados["disciplinas"]
        
        # 1. Matriz de Feromônio (τ): Inicia com um valor pequeno e constante.
        # Usar um DataFrame do pandas torna o acesso por ID muito mais fácil.
        self.matriz_feromonio = pd.DataFrame(1.0, index=professores, columns=disciplinas)

        # 2. Matriz de Informação Heurística (η): Baseada nas preferências.
        # Adicionamos um pequeno valor (0.1) para evitar que preferências 0 anulem
        # completamente a chance de uma escolha, permitindo mais exploração.
        self.info_heuristica = pd.DataFrame(self.dados_preparados["preferencias"]).T + 0.1
        
        print("Matrizes de feromônio e heurística inicializadas.")

    def _construir_solucao_formiga(self) -> tuple:
        """
        Simula uma única formiga construindo uma solução completa e válida.

        Returns:
            tuple: Contendo a solução (lista de tuplas (prof, disc)) e sua qualidade (soma das preferências).
        """
        #Constroi a solução da formiga
        solucao_formiga = []
        qualidade_solucao = 0
        cargas_atuais = {prof: 0 for prof in self.dados_preparados["professores"]}
       
        disciplinas_a_alocar = self.dados_preparados["disciplinas"].copy()
        np.random.shuffle(disciplinas_a_alocar)
        
        ch_max = self.dados_preparados["ch_max"]
        ch_disciplinas = self.dados_preparados["ch_disciplinas"]
        matriz_conflitos = self.dados_preparados["matriz_conflitos"]
        
        for disciplina in disciplinas_a_alocar:
            # 1. Encontrar professores candidatos (que não excedem a carga horária)
            candidatos = [p for p in self.dados_preparados["professores"] 
                          if (cargas_atuais[p] + ch_disciplinas[disciplina]) <= ch_max[p]]
            
            professores_com_conflito = set()
            for disciplina_conflitante in matriz_conflitos[matriz_conflitos[disciplina] == 1].index:
                # Encontra o professor que foi alocado para a disciplina conflitante (se já foi alocada)
                for prof, disc in solucao_formiga:
                    if disc == disciplina_conflitante:
                        professores_com_conflito.add(prof)
                        break
            candidatos = [p for p in candidatos if p not in professores_com_conflito]
            if not candidatos: # Se não houver candidatos, a solução é inviável
                return None, -1 # Retorna uma qualidade ruim

            # 2. Calcular a "atração" de cada candidato pela disciplina
            feromonio = self.matriz_feromonio.loc[candidatos, disciplina]
            heuristica = self.info_heuristica.loc[candidatos, disciplina]
            
            atratividade = (feromonio ** self.alfa) * (heuristica ** self.beta)
            
            # 3. Converter atratividade em probabilidades
            soma_atratividade = atratividade.sum()
            if soma_atratividade == 0: # Evita divisão por zero
                probabilidades = pd.Series(1/len(candidatos), index=candidatos)
            else:
                probabilidades = atratividade / soma_atratividade
            
            # 4. Escolher um professor com base nas probabilidades
            professor_escolhido = np.random.choice(probabilidades.index, p=probabilidades.values)
            
            # 5. Registrar a alocação e atualizar o estado
            solucao_formiga.append((professor_escolhido, disciplina))
            cargas_atuais[professor_escolhido] += ch_disciplinas[disciplina]
            qualidade_solucao += self.dados_preparados["preferencias"][professor_escolhido][disciplina]

        return solucao_formiga, qualidade_solucao

    def _atualizar_feromonio(self, solucoes_geracao: list):
        """
        Atualiza a matriz de feromônio com base nas soluções da geração atual.
        Inclui evaporação e depósito de novo feromônio.
        """
        # 1. Evaporação em toda a matriz
        self.matriz_feromonio *= (1 - self.taxa_evaporacao)

        # 2. Depósito de feromônio pela melhor formiga da geração
        if not solucoes_geracao:
            return
            
        # Encontra a melhor solução desta geração específica
        melhor_solucao_geracao = max(solucoes_geracao, key=lambda item: item[1])
        melhor_caminho, melhor_qualidade = melhor_solucao_geracao

        if melhor_caminho is None:
            return

        # A quantidade de feromônio depositada é proporcional à qualidade da solução
        deposito = 1.0 / (1 + (self.melhor_qualidade_global - melhor_qualidade)) if self.melhor_qualidade_global > 0 else 1.0

        for professor, disciplina in melhor_caminho:
            self.matriz_feromonio.loc[professor, disciplina] += deposito

    def _formatar_solucao_final(self):
        """
        Converte a melhor solução encontrada (lista de tuplas) em DataFrames
        para análise, seguindo o mesmo formato da saída da PLI.
        """
        if self.melhor_solucao_global is None:
            return None
        
        alocacoes = []
        for prof, disc in self.melhor_solucao_global:
            alocacoes.append({
                "id_disciplina": disc,
                "id_docente": prof,
                "preferencia": self.dados_preparados["preferencias"][prof][disc]
            })
        df_alocacao = pd.DataFrame(alocacoes)
        
        
        return {
            "alocacao_final": df_alocacao,
            "valor_objetivo": self.melhor_qualidade_global
        }

    def _resolver_nucleo(self):
        """
        Método principal que orquestra o processo de otimização ACO.
        """
        self._inicializar_parametros()

        for geracao in range(self.n_geracoes):
            solucoes_da_geracao = []
            
            for _ in range(self.n_formigas):
                solucao, qualidade = self._construir_solucao_formiga()
                if solucao:
                    solucoes_da_geracao.append((solucao, qualidade))
                    
                    # Verifica se esta formiga encontrou uma nova melhor solução global
                    if qualidade > self.melhor_qualidade_global:
                        self.melhor_qualidade_global = qualidade
                        self.melhor_solucao_global = solucao
            
            self._atualizar_feromonio(solucoes_da_geracao)

            if (geracao + 1) % 10 == 0: # Imprime o progresso a cada 10 gerações
                print(f"Geração {geracao + 1}/{self.n_geracoes} | Melhor Qualidade Global: {self.melhor_qualidade_global}")

        print("Otimização ACO concluída.")
        return self._formatar_solucao_final()
