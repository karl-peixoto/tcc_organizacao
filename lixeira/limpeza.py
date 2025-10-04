###FORMAS ANTERIORES DE LIDAR COM AS PREFERENCIAS DOS PROFESSORES

#TESTE COM SIMULAÇÃO
#Simulando as preferências
PESOS_PREFERENCIAS = [0.65, 0.15, 0.15, 0.05]
prof_ids_pref = []
disc_ids_pref = []
preferencias = []

# Loop para gerar a preferência de cada professor por cada disciplina
for prof_id in prof['id_docente']:
    for disc_id in dados['id_disciplina']:
        prof_ids_pref.append(prof_id)
        disc_ids_pref.append(disc_id)
        # Gera a preferência usando os pesos definidos
        pref = np.random.choice([0, 1, 2, 3], p=PESOS_PREFERENCIAS)
        preferencias.append(pref)

df_preferencias2 = pd.DataFrame({
    'id_docente': prof_ids_pref,
    'id_disciplina': disc_ids_pref,
    'preferencia': preferencias
})

df_preferencias.to_csv('../dados/preferencias.csv', index=False)



mapa_disciplinas = {
    'DELINEAMENTO E ANALISE DE EXPERIMENTOS 1': 'DELINEAMENTO E ANALISE DE EXPERIMENTOS',
    'COMPUTACAO EM ESTATISTICA 2 - R': 'COMPUTACAO EM ESTATISTICA 2: R',
    'COMPUTACAO EM ESTATISTICA 2 - PYTHON': 'COMPUTACAO EM ESTATISTICA 2: PYTHON',
    'TEORIA DE RESPOSTA AO ITEM': 'TEORIA DA RESPOSTA AO ITEM'
}
mapa_docentes = {
    'ALAN RICARDO RICARDO DA SILVA': 'ALAN RICARDO DA SILVA',
    'FELIPE SOUSA QUINTINO SOUSA QUINTINO': 'FELIPE SOUSA QUINTINO',
    'DEMERSON ANDRE POLLI ANDRE POLLI': 'DEMERSON ANDRE POLLI',
    'LUIZ FERNANDES CANCADO': 'ANDRE LUIZ FERNANDES CANCADO',
    'MONTEIRO DE CASTRO GOMES': 'EDUARDO MONTEIRO DE CASTRO GOMES'
}

df_preferencias = pd.read_csv('../dados/RespostaOfertaProfessores.csv',encoding='latin1',sep='#')
df_preferencias = df_preferencias.melt(id_vars=['Nome', 'Horários de preferência'], var_name='disciplina', value_name='preferencia')[['Nome', 'disciplina', 'preferencia']].rename(columns={'Nome':'docente'})
df_preferencias['disciplina'] = df_preferencias['disciplina'].str.upper().apply(unidecode).replace(mapa_disciplinas)
df_preferencias['docente'] = df_preferencias['docente'].str.upper().apply(unidecode).replace(mapa_docentes)

d1, d2 = df_preferencias.loc[df_preferencias['disciplina'].str.contains('TOPICOS')].copy(), df_preferencias.loc[df_preferencias['disciplina'].str.contains('TOPICOS')].copy()
d1['disciplina'] = 'TOPICOS EM ESTATISTICA 1'
d2['disciplina'] = 'TOPICOS EM ESTATISTICA 2'

d3 = df_preferencias.drop_duplicates(subset='disciplina').copy()
d3['docente'] = 'LEANDRO TAVARES CORREIA'
d3['preferencia'] = 'Em princípio, não tenho interesse.'

d4 = df_preferencias.drop_duplicates(subset='disciplina').copy()
d4['docente'] = 'ERITON BARROS DOS SANTOS'
d4['preferencia'] = 'Em princípio, não tenho interesse.'
d4.loc[d4['disciplina'] == 'ESTATISTICA APLICADA', 'preferencia'] = '1'



df_preferencias = df_preferencias[~df_preferencias['disciplina'].str.contains('TOPICOS')].copy()
df_preferencias = pd.concat([df_preferencias, d1,d2,d3,d4])

d5 = df_preferencias.drop_duplicates(subset='docente').copy()
d5['disciplina'] = 'TRABALHO DE CONCLUSAO DE CURSO 1'
d5['preferencia'] = '0'
d5.loc[d5['docente'] == 'LEANDRO TAVARES CORREIA', 'preferencia'] = '1'

df_preferencias = pd.concat([df_preferencias, d5])

df_preferencias = df_preferencias.merge(dados[['disciplina', 'id_disciplina']], on='disciplina',how='inner')
df_preferencias = df_preferencias.merge(prof[['docente', 'id_docente']], on='docente',how='left')

df_preferencias['preferencia'] = df_preferencias['preferencia'].apply(lambda x: 3 if x == '1' else 2 if x == '2' else 1 if x == '3' else 0)
df_preferencias[['id_docente', 'id_disciplina', 'preferencia']].to_csv('../dados/preferencias.csv',index=False)