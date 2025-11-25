## Link para visualização do projeto no Overleaf

[Link Visualização Overleaf](https://www.overleaf.com/read/yypgdjwsvdff#3ee062)


<h1 align="center">Alocador de Docentes – Versão V3</h1>

Versão simplificada do projeto de TCC (Departamento de Estatística / UnB) para gerar uma alocação única de docentes em disciplinas considerando preferências, cargas e conflitos. Foco: uso direto, rápido e transparente.

## 1. Visão Geral
O V3 mantém somente o essencial:
- Visualização inicial dos dados (professores e disciplinas)
- Execução síncrona dos algoritmos PLI, ACO e AG
- Fixação opcional de alocações (professor–disciplina) antes de otimizar
- Exportação do resultado em Excel

## 2. Pré-Requisitos
Python 3.9+ (recomendado) e pacotes:
```
Flask
pandas
numpy
PuLP
xlsxwriter
```
Instalação rápida:
```bash
pip install flask pandas numpy pulp xlsxwriter
```

## 3. Estrutura Essencial
```
projeto_aplicado/
	modelos/ (otimizadores e analisador)
	v3/
		app.py (aplicação Flask simplificada)
		templates/
			base.html
			dados_iniciais.html
			executar.html
			resultado.html
dados/ (CSV: docentes, disciplinas, preferencias, matriz_conflitos)
```

## 4. Executando
No diretório raiz:
```bash
python projeto_aplicado\v3\app.py
```
Acesse em seguida: `http://localhost:5000/`

## 5. Páginas
| Página | Descrição |
|--------|-----------|
| Dados Iniciais (`/`) | Lista professores (carga e contagem de preferências) e disciplinas (códigos / horários). |
| Executar (`/executar`) | Seleção de algoritmos, hiperparâmetros básicos e alocações fixas. |
| Resultado (`/resultado`) | Tabela compacta: professor, disciplina, preferência, código horário, código turma, carga e esfera de nível. Exporta Excel. |

## 6. Algoritmos e Parâmetros
| Algoritmo | Campo(s) configuráveis | Observação |
|-----------|------------------------|-----------|
| PLI | `Peso Penalidade (W)` | Penaliza violações / preferências ruins. |
| ACO | `n_formigas`, `n_geracoes`, `alfa`, `beta`, `taxa_evaporacao` | Heurística de construção probabilística. |
| AG | `n_populacao`, `n_geracoes`, `taxa_crossover`, `taxa_mutacao`, `tamanho_torneio`, `fator_penalidade` | Evolução com seleção por torneio. |

Seed (opcional) garante reprodutibilidade em ACO / AG (e qualquer lógica randômica).

## 7. Alocações Fixas
Na página Executar:
1. Clique em “Adicionar Alocação”.
2. Escolha professor e disciplina para cada linha.
3. Ao submeter, essas combinações entram pré-fixadas: aparecem com esfera roxa no resultado.
Validação interna impede exceder carga máxima do docente ao fixar.

## 8. Layout do Resultado
Colunas mostradas:
- Professor (nome + id)
- Disciplina (nome + código)
- Preferência (valor 0–3)
- Código Horário
- Código Turma
- CH da disciplina
- Nível (esfera colorida)

Esferas:
| Cor | Significado |
|-----|-------------|
| Verde | Preferência 3 |
| Amarelo | Preferência 2 |
| Laranja | Preferência 1 |
| Vermelho | Preferência 0 |
| Roxo | Alocação fixada |

Resumo superior mostra: escore total, contagem por nível e número de alocações fixadas.

## 9. Exportação
Botão “Exportar Excel” gera arquivo: `alocacao_YYYYMMDD_HHMMSS.xlsx` (download direto). Não há histórico salvo automaticamente.
