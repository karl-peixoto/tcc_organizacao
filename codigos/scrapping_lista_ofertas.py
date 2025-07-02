from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd


def get_html(departamento, nivel_ensino):
    """
    Acessa a página de oferta do SIGAA da UnB do semestre corrente, e extrai o conteúdo HTML das turmas do departamento especificado.
    Args:
        departamento (str): Nome do departamento (Conforme consta no Sigaa).
        nivel_ensino (str): Nível de ensino (Graduação, Mestrado, Doutorado, Stricto Senso).
    Returns:
        str: Conteúdo HTML da página com as turmas.
    """
    driver = webdriver.Chrome()
    driver.get("https://sigaa.unb.br/sigaa/public/turmas/listar.jsf")

    nivel = driver.find_element(By.ID, "formTurma:inputNivel")
    nivel.send_keys(nivel_ensino)


    unidade = driver.find_element(By.ID, "formTurma:inputDepto")
    unidade.send_keys(departamento)

    
    buscar = driver.find_element(By.XPATH, '//*[@id="formTurma"]/table/tfoot/tr/td/input[1]')
    buscar.click()


    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "listagem"))
    )

    
    html = driver.page_source
    driver.quit()
    return html

def extract_info(html):
    """
    Recebe o HTML da página do SIGAA e extrai as informações das turmas de graduação do Departamento de Estatística.
    Args:
        html (str): conteudo em HTML da pagina.
    Returns:
        pd.DataFrame: DataFrame com as infomracoes da oferta do semestre.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "listagem"})
    if not table:
        print("Tabela não encontrada!")
        return []

    turmas = []
    disciplina = None

    for row in table.find_all("tr"):
        if "agrupador" in row.get("class", []):
            disciplina = row.find("span", class_="tituloDisciplina")
            if disciplina:
                disciplina = disciplina.text.strip()
            else:
                disciplina = None
        
        
        elif "linhaPar" in row.get("class", []) or "linhaImpar" in row.get("class", []):
            cols = row.find_all("td")
            if len(cols) < 7:
                continue
            turma = {
                "codigo_disciplina": disciplina.split(' - ')[0].strip() if disciplina else None,
                "disciplina": disciplina.split(' - ')[1].strip() if disciplina else None,
                "codigo_turma": cols[0].text.strip(),
                "ano_periodo": cols[1].text.strip(),
                "docente": cols[2].text.strip().split('(')[0].strip(),
                "carga_horaria": cols[2].text.strip().split('(')[-1].replace(')', '').strip().replace('h', ''),
                "horario": cols[3].text.strip().split('\n')[0],
                "horario_extenso": cols[3].text.strip().split('\t')[-1],
                "vagas_ofertadas": cols[4].text.strip(),
                "vagas_ocupadas": cols[5].text.strip(),
                "local": cols[6].text.strip(),
            }
            turmas.append(turma)
    
    dados = pd.DataFrame(turmas)
    return dados

def main():
    departamento = "PROGRAMA DE PÓS-GRADUAÇÃO EM ESTATÍSTICA - BRASÍLIA"
    nivel_ensino = "STRICTO SENSO"

    html = get_html(departamento, nivel_ensino)
    dados = extract_info(html)

    periodo = dados['ano_periodo'][0].replace('.','_')

    if not dados.empty:
        print(f"{len(dados)} Turmas encontradas para o {departamento} no nível de ensino {nivel_ensino} no período {periodo}.")
        print(dados.head())
        dados.to_csv(f"dados/oferta_{periodo}_{departamento.lower().replace(' ', '-')}_{nivel_ensino.lower()}.csv", index=False)
    else:
        print("Nenhuma turma encontrada.")
    

if __name__ == "__main__":
    main()
