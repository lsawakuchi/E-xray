import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

'''
Gera as series de Ideb por ano para serem usadas na analise cluster que vou fazer
'''

class Preprocessing(object):
    def __init__(self, file_name,  data_type, encoding='utf-8'):
        self.file = file_name
        self.data_type = data_type
        self.encoding = encoding
        self.dados = None
        self.dados_temporais = None
    @property
    def campos_validos(self):
        if self.data_type == 'escola':
            return ['Co_UF', 'Cod_Escola_Completo', 'Rede', 'Ideb2005', 'Ideb2007', 'Ideb2009',
                    'Ideb2011', 'Ideb2013', 'Ideb2015', 'Ideb2017']
        else :
            if self.data_type == 'municipio':
                return ['Co_UF', 'Cod_Municipio_Completo', 'Rede', 'Ideb2005', 'Ideb2007', 'Ideb2009',
                    'Ideb2011', 'Ideb2013', 'Ideb2015', 'Ideb2017']
            else:
                return ['UF_REG', 'Rede', 'Ideb2005', 'Ideb2007', 'Ideb2009',
                    'Ideb2011', 'Ideb2013', 'Ideb2015', 'Ideb2017']

    def load_file(self):
        df = pd.read_csv(self.file, encoding=self.encoding)
        df = df[self.campos_validos].copy()
        if self.data_type == 'uf':
            df.columns = [self.data_type, 'tp_rede', 'i05', 'i07', 'i09', 'i11', 'i13', 'i15', 'i17']
        else :
            df.columns = ['uf',  self.data_type, 'tp_rede', 'i05', 'i07', 'i09', 'i11', 'i13', 'i15', 'i17']
            df[self.data_type] = df[self.data_type].astype(int)
        if self.dados is None:
            self.dados = df
    
    def fill_missing(self):
        lista_col = ['i05', 'i07', 'i09', 'i11', 'i13', 'i15', 'i17']
        for el in lista_col:
            self.dados[el] = self.dados.apply(lambda x : '0' if x[el]=='-' else x[el], axis=1)
            
    def formata_serie(self):
        resp = []
        for row in self.dados.iterrows():
            resp.append(row)
        serie = []
        for i in range(resp.__len__()):
            _row = dict(list(resp[i][1:])[0])
            _df = pd.DataFrame({"uf" : [_row.get("uf")]*7, 
                                 self.data_type : [_row.get(self.data_type)]*7, 
                                 "tp_rede" : [_row.get("tp_rede")]*7,
                                 "ano" : [2005, 2007, 2009, 2011, 2013, 2015, 2017],
                                "ideb" : [_row.get('i05'), _row.get('i07'), _row.get('i09'),
                                         _row.get('i11'), _row.get('i13'), _row.get('i15'), _row.get('i17')]})
            serie.append(_df)
        serie = pd.concat(serie)
        serie['ideb'] = serie.apply(lambda x : float(x['ideb']), axis=1)
        if self.data_type == 'uf':
            serie = serie.iloc[:, 1:]
        if self.dados_temporais is None:
            self.dados_temporais = serie
    
    def dados_finais(self):
        self.load_file()
        self.fill_missing()
        self.formata_serie()
        return self.dados_temporais
    
if __name__ == '__main__':
    pp = Preprocessing(file_name='ideb_uf_regioes_anosiniciais2005_2017.csv', data_type='uf', encoding = 'latin-1')
    resp2 = pp.dados_finais()
    
    