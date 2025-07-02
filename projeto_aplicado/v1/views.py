from flask import Blueprint, render_template
import pandas as pd

bp = Blueprint('main', __name__)
DIR_NAME = 'C:\\Users\\kmenezes\\OneDrive - unb.br\\tcc caralho'
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

