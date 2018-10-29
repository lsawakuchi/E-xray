import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from xray.cleaning.preprocessing import Preprocessing

class SeriesIdeb(object):
    def __init__(self, escola):
        self.escola = escola
        self.series_iniciais = None
        self.series_finais = None
        self.series_diretas = None
        self.corr_direta = None
        self.ind_disp = None
        self.series_indiretas = None
        self.corr_indireta = None
        self.tx_progressao = None
        self.tx_evolucao = None
        #self.metricas = None
        
    @property
    def anos_iniciais(self):
        return [2005, 2007, 2009, 2011, 2013]
    
    def load_series(self):
        if self.series_iniciais is None:
            pp = Preprocessing(file_name='ideb_escolas_anosiniciais2005_2017.csv', data_type='escola', encoding = 'latin-1')
            df0 = pp.dados_finais()
            df0.rename(columns={'ideb' : 'ideb_ai'}, inplace=True)
            self.series_iniciais = df0
            
        if self.series_finais is None:
            pp = Preprocessing(file_name='ideb_escolas_anosfinais2005_2017.csv', data_type='escola', encoding = 'latin-1')
            df1 = pp.dados_finais()
            df1.rename(columns={'ideb' : 'ideb_af'}, inplace=True)
            self.series_finais = df1
            
    ##seleciona a mesma janela temporal para as series de ideb inicial e final, para o calculo da correlacao direta  
    def seleciona_periodo(self):
        d0 = self.series_iniciais[self.series_iniciais['escola']==self.escola]
        d0 = d0[d0['ideb_ai']!=0]

        d1 = self.series_finais[self.series_finais['escola']==self.escola]
        d1 = d1[d1['ideb_af']!=0]

        final = d0[['escola', 'ano', 'ideb_ai']].merge(d1[['escola', 'ano', 'ideb_af']], left_on=['escola', 'ano'], right_on=['escola', 'ano'], how='left')
        final.dropna(inplace=True)
        
        if self.series_diretas is None:
            self.series_diretas = final


    ## calcula a correlacao entre as curvas de ideb inicial e final diretamente, ou seja desempenhos do inicio e do fim
    ## no mesmo ano de avaliacao. Métrica calculada somente para series de no minimo 3 observacoes consecutivas
    def comparacao_direta(self):
        df = self.series_diretas
        if df is None or df.shape[0] < 3:
            corr_direta = np.nan
        else: 
            corr_direta = df[['ideb_ai', 'ideb_af']].corr()['ideb_af'].iloc[0]
        if self.corr_direta is None:
            self.corr_direta = corr_direta
    
    def indice_disparidade(self):
        ind_disp =  -5*(1 - self.corr_direta) + 10
        if self.ind_disp is None:
            self.ind_disp = ind_disp
            
 ## gera a tabela de idebs do ano inicial e final para a mesma amostra de alunos avaliados
    
    def calcula_series_indiretas(self):
        lista_ini = self.anos_iniciais

        _df0 = self.series_iniciais[self.series_iniciais['escola']==self.escola]
        _df1 = self.series_finais[self.series_finais['escola']==self.escola]
        fr = []
        for yr in lista_ini:
            ano_fim = yr + 4
            dfini = _df0[_df0['ano']==yr]
            if dfini is None:
                return None
            dffim = _df1[_df1['ano']==ano_fim]
            if dffim is None:
                return None
            dfcomp = dfini.merge(dffim, left_on='escola', right_on='escola', how='left')
            dfcomp.rename(columns={'ano_x' : 'ano_ini', 'ano_y' : 'ano_fim'}, inplace=True)
            dfcomp = dfcomp[dfcomp['ideb_ai']!=0]
            dfcomp = dfcomp[dfcomp['ideb_af']!=0]
            dfcomp.dropna(inplace=True)
            dfcomp=dfcomp[['escola','tp_rede_x', 'uf_x', 'ano_ini', 'ano_fim', 'ideb_ai', 'ideb_af',]]
            dfcomp.rename(columns={'tp_rede_x' : 'tp_rede', 'uf_x' : 'uf'}, inplace=True)
            fr.append(dfcomp)
        if self.series_indiretas is None:
            self.series_indiretas = pd.concat(fr)
            
    # consideramos uma série de no mínimo 3 períodos de avaliacao para o calculo da correlacao indireta
    def correlacao_indireta(self):
        if self.series_indiretas is None or self.series_indiretas.shape[0] < 3:
            corr_indireta = np.nan
        corr_indireta = self.series_indiretas[['ideb_ai', 'ideb_af']].corr()['ideb_af'].iloc[0]
        if self.corr_indireta is None:
            self.corr_indireta = corr_indireta
    
    def calcula_progressao(self):
        _df = self.series_indiretas
        _df['prop'] = _df['ideb_af']/_df['ideb_ai']
        tx_progr =  _df['prop'].mean() - 1
        if self.tx_progressao is None:
            self.tx_progressao = tx_progr
        self.series_indiretas = _df
    
    def calcula_evolucao(self):
        _df = self.series_indiretas
        lista = _df['prop'].tolist()
        tx_evolucao = (lista[-1]-lista[0])/lista[0]
        if self.tx_evolucao is None:
            self.tx_evolucao = tx_evolucao

        
    def calcula(self):
        self.load_series()
        self.seleciona_periodo()
        self.comparacao_direta()
        self.indice_disparidade()
        self.calcula_series_indiretas()
        self.correlacao_indireta()
        self.calcula_progressao()
        self.calcula_evolucao()
        
if __name__ == '__main__':
    si = SeriesIdeb(escola = 53005740)
    si.calcula()          