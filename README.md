# Sistema de Alocação de Docentes em Disciplinas

Projeto de TCC (Departamento de Estatística / UnB) para otimização da alocação de docentes em disciplinas considerando preferências, carga máxima e conflitos, com interface web (Flask) e múltiplos algoritmos metaheurísticos e exatos.

## Objetivo
Gerar alocações que maximizam satisfação (preferências) e evitam atribuições penalizadas, utilizando diferentes abordagens:
- PLI (Programação Linear Inteira via PuLP)
- ACO (Ant Colony Optimization)
- AG (Algoritmo Genético)

## Principais Funcionalidades
- Dashboard de dados (professores, disciplinas e preferências)
- Execução síncrona de algoritmos múltiplos e comparação dos resultados
- Execução assíncrona com monitoramento de progresso e gráfico de convergência
- Histórico persistente de execuções (CSV) e detalhamento individual
- Simulações em lote (variação de parâmetros e seeds) com agregação estatística
- Validação de parâmetros com mensagens de erro e avisos
- Seed para reprodutibilidade

## Estrutura de Pastas (resumo)
- `projeto_aplicado/`
	- `modelos/`: Implementações dos otimizadores, analisador e persistência
	- `v2/`: Aplicação Flask (rotas, templates unificados, tarefas assíncronas, validação)
- `dados/`: Arquivos CSV (professores, disciplinas, preferências, conflitos, oferta)
- `codigos/`: Notebooks de análise e manipulação
- `extras/`, `imagens/`: Materiais suplementares

## Principais Módulos
- `otimizador_base.py`: Carregamento/filtragem de dados, recomposição de solução e seed
- `otimizador_pli.py`: Modelo exato (maximização preferências - penalidades W)
- `otimizador_aco.py`: Heurística baseada em feromônio e construção probabilística
- `otimizador_ag.py`: Evolução populacional (seleção por torneio, crossover, mutação)
- `analisador.py`: Métricas pós-solução (distribuição de preferências, escore total)
- `persistencia.py`: Registro em CSV (`historico_execucoes.csv` + alocações por ID)
- `tarefas.py`: Gerenciamento de jobs assíncronos e grupos (batch)
- `validacao.py`: Regras de sanity check para parâmetros

## Rotas da Aplicação (v2)
| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Dashboard (professores e disciplinas) |
| `/executar` | GET/POST | Formulário para execução síncrona de algoritmos |
| `/resultados` | GET | Exibe resultados da última execução síncrona |
| `/executar_async` | POST (JSON) | Dispara execução assíncrona de um algoritmo |
| `/progresso/<job_id>` | GET | Retorna status e métricas parciais de um job |
| `/historico` | GET | Lista execuções históricas filtráveis |
| `/historico/<id_execucao>` | GET | Detalhe de execução (alocação + métricas) |
| `/simulacoes` | GET/POST | Definição de lote (ranges, seeds, algoritmos) |
| `/simulacoes_progresso/<group_id>` | GET | Status agregado do lote |
| `/simulacoes_resultados/<group_id>` | GET | Tabela dos itens e estatísticas agregadas |

## Versão v3 Minimalista
Uma versão simplificada focada apenas em visualização inicial dos dados e execução direta dos algoritmos, sem persistência de histórico nem tarefas assíncronas.

### Diferenças Principais
- Sem CSV de histórico ou diretório de resultados automáticos.
- Execução sempre síncrona e imediata.
- Apenas páginas: Dados Iniciais, Executar, Resultado.
- Exportação do resultado para Excel (arquivo único) em memória.

### Rotas (v3)
| Rota | Método | Descrição |
|------|--------|-----------|
| `/` ou `/dados` | GET | Página de dados iniciais (preview preferências + resumo) |
| `/executar` | GET/POST | Formulário de seleção de algoritmo e seed opcional |
| `/resultado` | GET | Exibe última alocação otimizada (tabela colorida) |
| `/resultado/export` | GET | Exporta alocação atual para `.xlsx` |
| `/health` | GET | Status simples da aplicação |

### Colunas Exibidas
- `id_docente`, `id_disciplina`, `preferencia`, `ch_disciplina`, `ch_max`, `codigo_turma`, `horario_extenso`.
Se `codigo_turma` ou `horario_extenso` não existirem nos dados, são preenchidos com valores derivados ou placeholders.

### Esquema de Cores de Preferência
- 3: verde (`pref-3`)
- 2: amarelo (`pref-2`)
- 1: laranja (`pref-1`)
- 0: vermelho (`pref-0`)
- Alocações fixas (se futuramente usadas): roxo (`pref-fixada`)

### Executando v3
Instale dependências (se ainda não instaladas):
```bash
pip install flask pandas pulp numpy xlsxwriter
```
Inicie:
```bash
python projeto_aplicado\v3\app.py
```
Acesse: `http://localhost:5000/`

### Algoritmos Disponíveis (v3)
Seleção via formulário:
- `pli`: Programação Linear Inteira (exato)
- `aco`: Colônia de Formigas (parâmetros internos reduzidos)
- `ag`: Algoritmo Genético (parâmetros internos reduzidos)

### Exportação
O botão na página de resultado gera um arquivo Excel (`alocacao_YYYYMMDD_HHMMSS.xlsx`) para download imediato.

### Limitações Atuais da v3
- Não há comparação entre múltiplas execuções na interface.
- Não grava histórico nem permite simulações em lote.
- Não há monitoramento de progresso por geração (embora internamente heurísticas coletem métricas).

### Próximos Passos Possíveis
- Reintroduzir gráfico de convergência em uma aba opcional.
- Parametrização avançada dos hiperparâmetros direto no formulário.
- Cache estatístico agregado entre execuções (apenas em memória).


## Formato dos Resultados
Cada execução salva em CSV inclui (colunas principais):
- `id_execucao`, `timestamp`, `algoritmo`, `valor_objetivo`, `soma_preferencias`, `penalidade_total`, `num_alocacoes_preferencia_zero`, `tempo_execucao`, `seed`
- `metricas_iteracao_json`: lista JSON (por geração) contendo chaves como `geracao`, `melhor_global`, `melhor_geracao`, `media_geracao` (algoritmos heurísticos)
- Arquivo de alocação: `resultados/alocacao_<id_execucao>.csv`

## Execução Assíncrona
1. Via interface em `/executar` (botão "Executar Assíncrono")
2. Endpoint direto: 
```bash
curl -X POST http://localhost:5000/executar_async -H "Content-Type: application/json" -d '{"algoritmo":"aco","n_formigas":20,"n_geracoes":50,"seed":123}'
```
Response retorna `job_id`. Progresso:
```bash
curl http://localhost:5000/progresso/<job_id>
```
Estrutura de resposta inclui `status`, `progress` (0..1), `melhor_objetivo`, `metricas_ultimas`.

## Simulações em Lote
Formulário permite definir ranges (início, fim, passo) para parâmetros principais:
- PLI: `PENALIDADE_W`
- ACO / AG: `n_geracoes`
Além de lista de `seeds` (separadas por vírgula). Cada combinação gera um item no lote.
Status agregado em `/simulacoes_resultados/<group_id>` com estatísticas (média, melhor, pior, desvio).

## Validação de Parâmetros
Regras aplicadas antes da execução (síncrona, assíncrona e batch). Exemplos:
- `PENALIDADE_W > 0`
- `taxa_evaporacao` em `(0,1]`
- Taxas do AG (`crossover`, `mutacao`) em `(0,1]`
- Limites superiores para populações / gerações para evitar travamentos
Erros bloqueiam execução; avisos são apenas informativos.

## Instalação e Execução
Pré-requisitos: Python 3.x, dependências (pandas, pulp, flask, numpy, etc.). Instale:
```bash
pip install -r requirements.txt
```
Execute a aplicação (Windows CMD):
```bash
python projeto_aplicado\v2\app.py
```
Acesse: `http://localhost:5000/`

## Reprodutibilidade
Use o campo `Seed` nos formulários ou chave `seed` no JSON assíncrono para fixar aleatoriedade em ACO/AG (e qualquer lógica randômica futura).

## Melhorias Futuras
- Exportação direta de histórico / resultados (CSV/JSON) pela interface
- Métricas adicionais (fairness, balanceamento de carga)
- Boxplots e gráficos comparativos entre execuções de simulação
- Parametrização multi-parâmetro por algoritmo no lote (atualmente apenas 1 principal)
- Testes automatizados e cobertura

## Contribuição
Sugestões e melhorias são bem-vindas via issues ou pull requests.

## Licença
Uso acadêmico e interno. Verificar políticas institucionais antes de distribuição externa.
