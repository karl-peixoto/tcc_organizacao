from flask import Blueprint, render_template
import pandas as pd
from projeto_aplicado.modelos.otimizadores import OtimizadorPLI 

bp = Blueprint('main', __name__)
DIR_NAME = 'C:\\Users\\kmenezes\\OneDrive - unb.br\\tcc_organizacao'
# DataFrames de exemplo
consulta = pd.read_csv(f'{DIR_NAME}\\dados\\consulta_transformada.csv', encoding='utf-8-sig')
oferta = pd.read_csv(f'{DIR_NAME}\\dados\\oferta_transformada.csv', encoding='utf-8-sig')

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/consulta')
def show_consulta():
    return render_template('consulta.html', table=consulta.to_html(classes='data', index=False, border=0))

@bp.route('/oferta')
def show_oferta():
    return render_template('oferta.html', table=oferta.to_html(classes='data', index=False, border=0))

@bp.route('/otimizar_pli')
def rota_otimizar():
    """
    Esta rota será acionada quando o usuário clicar no link/botão para otimizar.
    """
    # Define a configuração (pode vir de um arquivo de config do Flask)
    config = {
        "ARQUIVOS_DADOS": {
            "disciplinas": "../dados/disciplinas.csv",
            "professores": "../dados/docentes.csv",
            "preferencias": "../dados/preferencias.csv",
        },
        "PENALIDADE_W": 4.0,
    }

    # Instancia e resolve, exatamente como no script anterior
    otimizador = OtimizadorPLI(config=config)
    resultados = otimizador.resolver()

    # Passa os DataFrames da solução para o template HTML renderizar
    return render_template('resultado.html', resultados=resultados)