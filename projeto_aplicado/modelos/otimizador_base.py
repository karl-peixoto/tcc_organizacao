import pandas as pd
from pathlib import Path
import os
import random
import numpy as np

class Otimizador:
    """
    Classe base para todos os otimizadores.
    Contém a lógica compartilhada de carregamento e preparação de dados.
    """
    def __init__(self, config: dict):
        """
        Inicializa o otimizador com as configurações e carrega/prepara os dados.
        """
        self.config = config
        self.dados_brutos = None
        self.dados_preparados = None
        self.alocacoes_fixas = self.config.get("ALOCACOES_FIXAS", [])
        self.pref_servico = self.config.get("PREFERENCIA_SERVICO", 3)
        self.pref_demais = self.config.get("PREFERENCIA_DEMAIS", 0)
        self.seed = self.config.get("SEED")

        if self.seed is not None:
            try:
                random.seed(self.seed)
                np.random.seed(self.seed)
                #print(f"Seed definida: {self.seed}")
            except Exception as e:
                #print(f"lhs_global[lhs_global['simulacao'] > 100]lha ao definir seed {self.seed}: {e}")
                pass

        #Atributo para guardar a parte já resolvida da solução
        self.solucao_parcial_fixa = None
        
        # Se recebermos DataFrames injetados no config, usamos diretamente e pulamos leitura de CSV.
        dados_injetados = self.config.get("DADOS_INJETADOS")
        if dados_injetados is not None:
            self.dados_brutos = dados_injetados
        else:
            self._carregar_dados()
        self._preparar_dados()
        #print(f"Otimizador base inicializado. {len(self.alocacoes_fixas)} alocações foram fixadas.")

    def _carregar_dados(self):
        """Método para carregar os dados dos arquivos CSV."""
        # Se já existir dados_brutos (por injeção), não recarregar
        if self.dados_brutos is not None:
            return
        try:
            caminho_raiz = Path(__file__).parent.parent.parent
            caminho_pasta_dados = caminho_raiz / "dados"

            #print(f"Buscando dados no diretório: {caminho_pasta_dados}")
            self.dados_brutos = {
                nome: pd.read_csv(caminho_pasta_dados / nome_arquivo)
                for nome, nome_arquivo in self.config.get("ARQUIVOS_DADOS", {}).items()
            }
            #print("Dados brutos carregados.")
        except FileNotFoundError as e:
            #print(f"Erro: Arquivo não encontrado - {e}.")
            raise

    def _normalizar_conflitos(self, df_conflitos_raw: pd.DataFrame, df_disciplinas: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza a matriz de conflitos para:
        - índice e colunas = lista completa de id_disciplina
        - remove coluna fantasma 'Unnamed: 0'
        - garante simetria e diagonal zero
        - tipo int (0/1)
        """
        df = df_conflitos_raw.copy()

        # Se vier com coluna de índice salva
        if 'Unnamed: 0' in df.columns and df.index.name != 'id_disciplina':
            df.set_index('Unnamed: 0', inplace=True)
        # Se a primeira coluna for id_disciplina explícita
        elif df.columns[0] == 'id_disciplina' and df.index.name != 'id_disciplina':
            df.set_index(df.columns[0], inplace=True)

        disciplinas = df_disciplinas['id_disciplina'].tolist()

        # Reindexa linhas e colunas
        df = df.reindex(index=disciplinas, columns=disciplinas)

        # Preenche ausentes
        df = df.fillna(0)

        # Força numérico 0/1
        df = (df.astype(int) > 0).astype(int)

        # Força simetria e diagonal zero
        df.values[np.diag_indices_from(df)] = 0
        df = ((df + df.T) > 0).astype(int)
        return df

    def _preparar_preferencias(self, df_professores, df_disciplinas, df_preferencias):
        """Prepara a matriz completa de preferências, tratando valores ausentes."""
        
        # Lógica de preparação
        lista_professores_total = df_professores['id_docente'].tolist()
        todas_as_disciplinas = df_disciplinas['id_disciplina'].tolist()
        disciplinas_servico = df_disciplinas[df_disciplinas['tipo_disciplina'] == 'SERVICO']['id_disciplina'].tolist()
        
           
        # 1. Identifica a lista de professores que efetivamente responderam ao formulário
        professores_que_responderam = df_preferencias['id_docente'].unique().tolist()
        #print(f"{len(professores_que_responderam)} de {len(lista_professores_total)} professores responderam ao formulário.")

        # 2. Pivota o DataFrame de preferências como antes.
        df_prefs_pivot = df_preferencias.pivot(
            index='id_docente',
            columns='id_disciplina',
            values='preferencia'
        )

        # 3. Reindexa o DataFrame para garantir que todas as disciplinas e todos os professores estejam presentes.
        #    Neste ponto, professores que não responderam terão suas linhas preenchidas com NaN.
        #    Professores que responderam mas pularam uma disciplina também terão NaN naquela célula.
        df_prefs_completo = df_prefs_pivot.reindex(
            index=lista_professores_total, 
            columns=todas_as_disciplinas
        )

        # 4. Aplica a regra de negócio:
        for disciplina_id in df_prefs_completo.columns:
            # Define o valor de preenchimento com base no tipo da disciplina
            if disciplina_id in disciplinas_servico:
                valor_preenchimento = self.pref_servico
            else:
                valor_preenchimento = self.pref_demais
        
            # Preenche todos os NaNs naquela coluna com o valor definido
            df_prefs_completo[disciplina_id] = df_prefs_completo[disciplina_id].fillna(valor_preenchimento)
        

        # Garante que todos os valores sejam inteiros
        df_prefs_completo = df_prefs_completo.astype(int)

        return df_prefs_completo


    def _preparar_dados(self):
        """Método para preparar os dados para a modelagem."""
        if self.dados_brutos is None:
            self._carregar_dados()
        
        
        df_professores = self.dados_brutos["professores"]
        df_disciplinas = self.dados_brutos["disciplinas"]
        df_preferencias = self.dados_brutos["preferencias"]
        df_conflitos = self.dados_brutos["conflitos"]

        df_conflitos = self._normalizar_conflitos(df_conflitos, df_disciplinas)
        
        # Lógica de preparação
        lista_professores_total = df_professores['id_docente'].tolist()
        todas_as_disciplinas = df_disciplinas['id_disciplina'].tolist()
        
        
        # NOVO MODELO DE CAPACIDADE: agora usamos número de disciplinas em vez de carga horária.
        # ch_max passa a representar o limite de disciplinas por professor (coluna max_disciplinas).
        # ch_disciplinas passa a ser custo unitário (1 disciplina = 1 unidade de capacidade).
        ch_max = df_professores.set_index('id_docente')['max_disciplinas'].to_dict()
        ch_disciplinas = {d: 1 for d in df_disciplinas['id_disciplina']}
        #df_conflitos.set_index(df_conflitos.columns[0], inplace=True)
        matriz_conflitos = df_conflitos

        # --- Etapa 1: Processar e Isolar Alocações Fixas ---
        cargas_atuais = {prof: 0 for prof in lista_professores_total}
        disciplinas_fixadas = set()
        alocacoes_resolvidas = []
        for alocacao in self.alocacoes_fixas:
            prof = alocacao['professor']
            disc = alocacao['disciplina']
            
            # Validação
            carga_futura = cargas_atuais[prof] + ch_disciplinas[disc]
            if carga_futura > ch_max[prof]:
                raise ValueError(f"Erro: Alocações fixas para {prof} excedem seu limite de disciplinas!")
            
            cargas_atuais[prof] = carga_futura
            disciplinas_fixadas.add(disc)
            alocacoes_resolvidas.append({"id_disciplina": disc, "id_docente": prof})

        # Armazena a parte da solução que já está pronta
        self.solucao_parcial_fixa = pd.DataFrame(alocacoes_resolvidas)
        #print(f"{len(self.solucao_parcial_fixa)} alocações fixas validadas e separadas.")


        # --- Etapa 2: Filtrar Professores e Disciplinas para criar o problema reduzido ---
        disciplinas_a_alocar = [d for d in todas_as_disciplinas if d not in disciplinas_fixadas]
    
        # Professores que ainda têm capacidade (número de disciplinas) disponível
        professores_a_considerar = [
            p for p, ch in cargas_atuais.items() 
            if ch < ch_max[p]
        ]

        # --- Etapa 3: Preparar os Dados Reduzidos para o Otimizador ---
        
        # Filtra preferências (lógica de preenchimento de NaN permanece)
        self.preferencias_completas = self._preparar_preferencias(df_professores, df_disciplinas, df_preferencias)
        
        # Agora filtra o DataFrame de preferências para o problema reduzido
        df_prefs_reduzido = self.preferencias_completas.loc[professores_a_considerar, disciplinas_a_alocar]
        prefs_reduzidas = df_prefs_reduzido.to_dict(orient='index')

        # Filtra as demais estruturas de dados
        ch_remanescente = {p: ch_max[p] - cargas_atuais[p] for p in professores_a_considerar}
        ch_disciplinas_reduzido = {d: ch for d, ch in ch_disciplinas.items() if d in disciplinas_a_alocar}
        
        matriz_conflitos_reduzida = matriz_conflitos.loc[disciplinas_a_alocar, disciplinas_a_alocar]

        # Pré-computar estrutura leve de conflitos (adjacência) para acelerar consultas
        # Formato: {disciplina: set(disciplinas_em_conflito)}
        conflito_adj = {}
        if isinstance(matriz_conflitos_reduzida, pd.DataFrame):
            for d in matriz_conflitos_reduzida.index:
                linha = matriz_conflitos_reduzida.loc[d]
                conflito_adj[d] = set(linha.index[linha.values.astype(int) == 1])


       # Armazena o problema final e reduzido que será visto pelos filhos
        self.dados_preparados = {
            "professores": professores_a_considerar,
            "disciplinas": disciplinas_a_alocar,
            "ch_max": ch_remanescente,
            "ch_disciplinas": ch_disciplinas_reduzido,
            "preferencias": prefs_reduzidas,
            "matriz_conflitos": matriz_conflitos_reduzida,
            "conflitos_adj": conflito_adj
        }
        #print("Dados preparados. O problema foi reduzido para a otimização.")

    def set_dados_brutos(self, dados: dict):
        """Permite injetar DataFrames já prontos e reprocesar a preparação."""
        self.dados_brutos = dados
        self._preparar_dados()

    def resolver(self, callback_iteracao=None):
        """Executa otimização completa.
        callback_iteracao: função opcional chamada a cada iteração/geração pelos filhos.
        """
        #print("\n--- Iniciando Processo de Otimização ---")
        solucao_otimizada_parcial = self._resolver_nucleo(callback_iteracao=callback_iteracao)
        
        if solucao_otimizada_parcial is None:
            #print("Otimizador não encontrou uma solução.")
            self.resultados = None
            return None

        solucao_final = self._recompor_solucao(solucao_otimizada_parcial)
        self.resultados = solucao_final
        #print("--- Processo de Otimização Concluído ---")
        return self.resultados
   
    def _resolver_nucleo(self, callback_iteracao=None):
        """Método 'abstrato'. CADA FILHO DEVE IMPLEMENTAR ESTE MÉTODO."""
        raise NotImplementedError("Este método deve ser implementado pela subclasse.")

    def _recompor_solucao(self, solucao_parcial: dict):
        """Junta a solução fixa com a solução parcial do otimizador."""
        df_parcial = solucao_parcial.get("alocacao_final")
        
        # Pega as preferências e CHs totais para enriquecer os DFs antes de concatenar
        df_prefs_total = self.preferencias_completas
        ch_total = self.dados_brutos["disciplinas"].set_index('id_disciplina')['carga_horaria']

        # Garante que o DF de solução fixa tenha as mesmas colunas que o DF parcial
        df_fixo_enriquecido = self.solucao_parcial_fixa.copy()
        if not df_fixo_enriquecido.empty:
            df_fixo_enriquecido['preferencia'] = df_fixo_enriquecido.apply(
                lambda row: df_prefs_total.loc[row['id_docente'], row['id_disciplina']], axis=1
            )
            df_fixo_enriquecido['ch_disciplina'] = df_fixo_enriquecido['id_disciplina'].map(ch_total)

        if df_parcial is None or df_parcial.empty:
            df_completo = df_fixo_enriquecido
        else:
            df_completo = pd.concat([df_fixo_enriquecido, df_parcial], ignore_index=True)
        
        # Mantém outras chaves do resultado parcial (valor_objetivo, metricas_iteracao, seed, etc.)
        resultado_final = {k: v for k, v in solucao_parcial.items() if k != "alocacao_final"}
        resultado_final["alocacao_final"] = df_completo
        if "seed" not in resultado_final:
            resultado_final["seed"] = self.seed
        return resultado_final

