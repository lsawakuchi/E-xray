import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

'''
Gera as series de Ideb por ano para serem usadas na analise cluster que vou fazer
'''

class Preprocessing(object):
    def __init__(self, file_name, encoding='utf-8'):
        self.file = file_name
        self.encoding = encoding
        self.dados = None
        self.dados_temporais = None
    @property
    def campos_validos(self):
        return ['Co_UF', 'Cod_Municipio_Completo', 'Cod_Escola_Completo', 'Rede', 'Ideb2005', 'Ideb2007', 'Ideb2009',
                'Ideb2011', 'Ideb2013', 'Ideb2015', 'Ideb2017']
    def load_file(self):
        df = pd.read_csv(self.file, encoding=self.encoding)
        df = df[self.campos_validos].copy()
        df.columns = ['uf', 'cod_mun', 'cod_escola', 'tp_rede', 'i05', 'i07', 'i09', 'i11', 'i13', 'i15', 'i17']
        df['cod_escola'] = df['cod_escola'].astype(int)
        df['cod_mun'] = df['cod_mun'].astype(int)
        if self.dados is None:
            self.dados = df
    
    def fill_missing(self):
        lista_col = ['i05', 'i07', 'i09', 'i11', 'i13', 'i15', 'i17']
        for el in lista_col:
            self.dados[el] = self.dados.apply(lambda x : '0' if x[el]=='-' else x[el], axis=1)
    
    def formata_serie(self):
        frames = []
        for escola in self.dados['cod_escola'].tolist():
            dfesc = self.dados[self.dados['cod_escola'] == escola]
            _df = dfesc.iloc[:, 4:].T
            _df.columns = ['ideb']
            lista_ideb = _df['ideb'].tolist()
            frames.append(pd.DataFrame({'cod_escola' : [dfesc['cod_escola'].iloc[0]]*7,
                                       'cod_mun' : [dfesc['cod_mun'].iloc[0]]*7,
                                       'uf' : [dfesc['uf'].iloc[0]]*7,
                                       'tp_rede' : [dfesc['tp_rede'].iloc[0]]*7,
                                       'ano' : [2005, 2007, 2009, 2011, 2013, 2015, 2017],
                                       'ideb' : lista_ideb}))
        resp = pd.concat(frames)
        resp['ideb'] = resp.apply(lambda x : float(x['ideb']), axis=1)
        if self.dados_temporais is None:
            self.dados_temporais = resp
    
    def dados_finais(self):
        self.load_file()
        self.fill_missing()
        self.formata_serie()
        return self.dados_temporais
    
if __name__ == '__main__':
    pp = Preprocessing('ideb_escolas_anosfinais2005_2017.csv', 'latin-1')
    resp2 = pp.dados_finais()
    