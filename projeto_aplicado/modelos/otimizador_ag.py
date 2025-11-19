import pandas as pd
import numpy as np
import random
from .otimizador_base import Otimizador

class OtimizadorAG(Otimizador):
    """
    Implementação do otimizador de alocação de docentes usando Algoritmos Genéticos (AG).
    
    Herda a lógica de carregamento e preparação de dados da classe Otimizador.
    """
    def __init__(self, config: dict):
        """
        Inicializa o otimizador AG.

        Args:
            config (dict): Dicionário de configuração contendo:
                - ARQUIVOS_DADOS: Caminhos para os CSVs.
                - AG_PARAMS: Dicionário com os hiperparâmetros do AG:
                    - n_populacao (int): Tamanho da população.
                    - n_geracoes (int): Número de gerações (iterações).
                    - taxa_crossover (float): Probabilidade de ocorrer crossover.
                    - taxa_mutacao (float): Probabilidade de um gene sofrer mutação.
                    - tamanho_torneio (int): Número de indivíduos no torneio de seleção.
                    - fator_penalidade (int): Grandeza da penalidade por violar restrições.
        """
        # 1. Chama o __init__ da classe mãe para carregar e preparar os dados
        super().__init__(config)
        
        # 2. Atributos específicos do AG
        params = self.config.get("AG_PARAMS", {})
        self.n_populacao = params.get("n_populacao", 50)
        self.n_geracoes = params.get("n_geracoes", 200)
        self.taxa_crossover = params.get("taxa_crossover", 0.8)
        self.taxa_mutacao = params.get("taxa_mutacao", 0.02)
        self.tamanho_torneio = params.get("tamanho_torneio", 3)
        self.fator_penalidade = params.get("fator_penalidade", 1000)

        # Atributos que serão preenchidos durante a execução
        self.populacao = []
        self.melhor_solucao_global = None
        self.melhor_fitness_global = -float('inf')
        self.metricas_iteracao = []

        self.disciplinas_do_problema = self.dados_preparados["disciplinas"]
        self.map_disc_idx = {disc_id: i for i, disc_id in enumerate(self.disciplinas_do_problema)}
    
        
        print(f"OtimizadorAG pronto.")

    def _escolher_professor_valido(self, disciplina_id: str, solucao_parcial: dict, cargas_atuais: dict) -> str:
        """
        Escolhe um professor para uma disciplina, respeitando as restrições e
        usando as preferências como peso para um sorteio.
        """
        ch_max = self.dados_preparados["ch_max"]
        ch_disciplinas = self.dados_preparados["ch_disciplinas"]
        matriz_conflitos = self.dados_preparados["matriz_conflitos"]
        
        # 1. Filtra por carga horária
        candidatos = [p for p in self.dados_preparados["professores"] 
                      if (cargas_atuais.get(p, 0) + ch_disciplinas[disciplina_id]) <= ch_max[p]]
        
        # 2. Filtra por conflito de horário
        disciplinas_com_conflito = matriz_conflitos[matriz_conflitos[disciplina_id] == 1].index
        
        professores_invalidos = set()
        for prof, disc_alocada in solucao_parcial.items():
            if disc_alocada in disciplinas_com_conflito:
                professores_invalidos.add(prof)

        candidatos_validos = [p for p in candidatos if p not in professores_invalidos]
        
        # Fallback: se nenhuma opção válida for encontrada, sorteia de todos
        if not candidatos_validos:
            return random.choice(self.dados_preparados["professores"])

        # 3. Sorteio Ponderado pela Preferência
        preferencias = [self.dados_preparados["preferencias"][p][disciplina_id] for p in candidatos_validos]
        
        soma_prefs = sum(preferencias)
        if soma_prefs == 0:
            # Se todos os candidatos têm preferência 0, faz um sorteio normal
            return random.choice(candidatos_validos)
        else:
            # Realiza o sorteio ponderado
            return random.choices(candidatos_validos, weights=preferencias, k=1)[0]


    def _gerar_populacao_inicial(self):
        """Cria a população inicial de soluções (cromossomos) aleatórias."""
        
        # 1. Gera a população
        for _ in range(self.n_populacao):
           # Dicionário para construir a solução passo a passo {disciplina: professor}
            solucao_parcial_dict = {} 
            cargas_atuais = {prof: 0 for prof in self.dados_preparados["professores"]}
            
            disciplinas_shuffled = self.disciplinas_do_problema.copy()
            random.shuffle(disciplinas_shuffled)
            
            for disciplina in disciplinas_shuffled:
                professor_escolhido = self._escolher_professor_valido(disciplina, solucao_parcial_dict, cargas_atuais)
                solucao_parcial_dict[disciplina] = professor_escolhido
                cargas_atuais[professor_escolhido] += self.dados_preparados["ch_disciplinas"][disciplina]
            
            # Converte o dicionário para a lista (cromossomo) na ordem correta
            cromossomo = [solucao_parcial_dict[disc] for disc in self.disciplinas_do_problema]
            self.populacao.append(cromossomo)
            
        print(f"População inicial com {self.n_populacao} indivíduos criada, respeitando os genes fixos.")


    def _calcular_fitness(self, cromossomo: list) -> float:
        """
        Calcula a pontuação de fitness de um único cromossomo.
        A pontuação é a soma das preferências, com uma alta penalidade
        se a restrição de carga horária for violada.
        """
        score_preferencias = 0
        cargas_atuais = {prof: 0 for prof in self.dados_preparados["professores"]}
        
        # Calcula score e carga horária
        for idx_disciplina, id_professor in enumerate(cromossomo):
            id_disciplina = self.dados_preparados["disciplinas"][idx_disciplina]
            score_preferencias += self.dados_preparados["preferencias"][id_professor][id_disciplina]
            cargas_atuais[id_professor] += self.dados_preparados["ch_disciplinas"][id_disciplina]

        # Calcula a penalidade
        excesso_total = 0
        for id_professor, carga_atribuida in cargas_atuais.items():
            carga_maxima = self.dados_preparados["ch_max"][id_professor]
            if carga_atribuida > carga_maxima:
                excesso_total += (carga_atribuida - carga_maxima)

        # Penaliza conflitos de horario
        matriz_conflitos = self.dados_preparados["matriz_conflitos"]
        numero_de_conflitos = 0
        
        # Mapeia quais disciplinas cada professor recebeu neste cromossomo
        alocacoes_por_prof = {prof: [] for prof in self.dados_preparados["professores"]}
        for idx_disciplina, id_professor in enumerate(cromossomo):
            id_disciplina = self.dados_preparados["disciplinas"][idx_disciplina]
            alocacoes_por_prof[id_professor].append(id_disciplina)

        # Verifica os conflitos para cada professor
        for prof, disciplinas_do_prof in alocacoes_por_prof.items():
            # Só pode haver conflito se o professor tiver 2 ou mais disciplinas
            if len(disciplinas_do_prof) > 1:
                # Compara cada par de disciplinas do professor, sem repetir
                for i in range(len(disciplinas_do_prof)):
                    for j in range(i + 1, len(disciplinas_do_prof)):
                        d1 = disciplinas_do_prof[i]
                        d2 = disciplinas_do_prof[j]
        
                        if matriz_conflitos.loc[d1, d2] == 1:
                            numero_de_conflitos += 1
        
        penalidade_carga = self.fator_penalidade * excesso_total
        penalidade_conflito = self.fator_penalidade * numero_de_conflitos
        
        fitness = score_preferencias - penalidade_carga - penalidade_conflito
        return fitness

    def _selecao_por_torneio(self, fitness_populacao: list) -> list:
        """Seleciona um indivíduo da população usando o método do torneio."""
        selecionados = random.sample(list(enumerate(fitness_populacao)), self.tamanho_torneio)
        # Encontra o indivíduo com o maior fitness dentro do grupo selecionado
        vencedor = max(selecionados, key=lambda item: item[1])
        return self.populacao[vencedor[0]]

    def _crossover(self, pai1: list, pai2: list) -> tuple:
        """Realiza o crossover de ponto único entre dois pais para gerar dois filhos."""
        if random.random() > self.taxa_crossover:
            return pai1, pai2 # Retorna os pais sem alteração

        ponto_corte = random.randint(1, len(pai1) - 1)
        filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
        filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
        return filho1, filho2

    def _mutacao(self, cromossomo: list) -> list:
        """Aplica uma mutação 'inteligente' em um gene do cromossomo."""
        cromossomo_mutado = cromossomo[:]
        for i in range(len(cromossomo_mutado)):
            if random.random() < self.taxa_mutacao:
                disciplina_a_mutar = self.disciplinas_do_problema[i]
                
                # Recria o estado parcial do cromossomo (sem a disciplina a ser mutada)
                solucao_parcial_dict = {}
                cargas_atuais = {prof: 0 for prof in self.dados_preparados["professores"]}
                for j in range(len(cromossomo_mutado)):
                    if i == j: continue # Pula o gene que será mutado
                    
                    prof_atual = cromossomo_mutado[j]
                    disc_atual = self.disciplinas_do_problema[j]
                    solucao_parcial_dict[prof_atual] = disc_atual
                    cargas_atuais[prof_atual] += self.dados_preparados["ch_disciplinas"][disc_atual]

                # Escolhe um novo professor válido para a disciplina
                novo_professor = self._escolher_professor_valido(disciplina_a_mutar, solucao_parcial_dict, cargas_atuais)
                cromossomo_mutado[i] = novo_professor
        return cromossomo_mutado

    def _formatar_solucao_final(self):
        """Converte a melhor solução encontrada em DataFrames para análise."""
        if self.melhor_solucao_global is None:
            return None
        
        alocacoes = []
        for idx_disc, id_prof in enumerate(self.melhor_solucao_global):
            id_disc = self.dados_preparados["disciplinas"][idx_disc]
            alocacoes.append({
                "id_disciplina": id_disc,
                "id_docente": id_prof,
                "preferencia": self.dados_preparados["preferencias"][id_prof][id_disc]
            })
        df_alocacao = pd.DataFrame(alocacoes)
        
        return {
            "alocacao_final": df_alocacao,
            "valor_objetivo": self.melhor_fitness_global
        }

    def _resolver_nucleo(self, callback_iteracao=None):
        """Resolve AG; chama callback_iteracao por geração se fornecido."""
        self._gerar_populacao_inicial()

        for geracao in range(self.n_geracoes):
            # 1. Avaliação
            fitness_populacao = [self._calcular_fitness(ind) for ind in self.populacao]

            # 2. Acompanhamento do melhor indivíduo
            melhor_fitness_geracao = max(fitness_populacao)
            if melhor_fitness_geracao > self.melhor_fitness_global:
                self.melhor_fitness_global = melhor_fitness_geracao
                idx_melhor = fitness_populacao.index(melhor_fitness_geracao)
                self.melhor_solucao_global = self.populacao[idx_melhor].copy()

            # 3. Seleção e Reprodução
            nova_populacao = []
            while len(nova_populacao) < self.n_populacao:
                pai1 = self._selecao_por_torneio(fitness_populacao)
                pai2 = self._selecao_por_torneio(fitness_populacao)
                
                filho1, filho2 = self._crossover(pai1, pai2)
                
                nova_populacao.append(self._mutacao(filho1))
                nova_populacao.append(self._mutacao(filho2))
            
            self.populacao = nova_populacao[:self.n_populacao] # Garante o tamanho da população

            # Coleta métricas da geração
            melhor_geracao = melhor_fitness_geracao
            media_geracao = float(np.mean(fitness_populacao)) if fitness_populacao else None
            self.metricas_iteracao.append({
                "geracao": geracao + 1,
                "melhor_geracao": melhor_geracao,
                "melhor_global": self.melhor_fitness_global,
                "media_geracao": media_geracao
            })

            if (geracao + 1) % 20 == 0:
                print(f"Geração {geracao + 1}/{self.n_geracoes} | Melhor Global: {self.melhor_fitness_global:.2f}")

        print("Otimização AG concluída.")
        resultado = self._formatar_solucao_final()
        if resultado is None:
            return None
        resultado["metricas_iteracao"] = self.metricas_iteracao
        return resultado

