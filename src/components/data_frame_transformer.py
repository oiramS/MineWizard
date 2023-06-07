from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import numpy as np 
import pandas as pd
class Df_transformer:
    def __init__(self, dataframe=None):
      self.dataframe = dataframe
      self.varianza = None
      self.varianza_acumulada = None
      
      
    def set_dataframe(self, dataframe):
        self.dataframe = dataframe
        
    def set_cur_scaler(self, scaler):
        self.cur_scaler = scaler
        
    def set_cur_mscaler(self, mscaler):
        self.cur_mscaler = mscaler

    def set_dataframe_PCA(self, dataframe):
        self.dataframe = dataframe
        self.standar_scaler = StandardScaler().fit_transform(dataframe)
        self.minmax_scaler = MinMaxScaler().fit_transform(dataframe)
        
    def set_varianza(self, varianza):
        self.varianza = varianza
    
    def set_predictor(self, predictor):
        self.predictor = predictor
    
    def set_estimators(self, estimators):
        self.estimators = estimators
    
    def set_feature_columns(self, feature_columns):
        self.feature_columns = feature_columns
        
    def set_kmdf(self, kdata_frame):
        self.kdata_frame = kdata_frame
    
    def get_kmdf(self):
        return self.kdata_frame
    
    def get_feature_columns(self):
        return self.feature_columns 
    
    def get_estimators(self):
        return self.estimators
    
    def get_preditor(self):
        return self.predictor
    
    def get_standar_scaler(self):
        return self.standar_scaler
    
    def get_minmax_scaler(self):
        return self.minmax_scaler
    
    def get_cur_scaler(self):
        return self.cur_scaler
    
    def get_cur_mscaler(self):
        return self.cur_mscaler
    
    def get_df(self):
        return self.dataframe
    
    def get_df_numeric(self):
        return self.dataframe.select_dtypes(include=np.number)
    
    def get_varianza(self):
        return self.varianza
    
    def get_varianza_acum(self, value):
        if any(self.varianza != None):
            self.varianza_acumulada= sum(self.varianza[0:value])
            return self.varianza_acumulada
        return None
    
    def get_info_as_df(self):
        index_ = self.dataframe.columns
        info = self.dataframe.dtypes.to_frame('dtypes').astype(str)
        info['column_name'] = index_
        info['non_null'] = self.dataframe.count().astype(str)
        info['unique_values'] = self.dataframe.apply(lambda srs: len(srs.unique())).astype(str)
        return info[['column_name', 'dtypes', 'non_null', 'unique_values']]