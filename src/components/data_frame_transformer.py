from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

class Df_transformer:
    def __init__(self, dataframe=None):
      self.dataframe = dataframe
      self.varianza = None
      self.varianza_acumulada = None
      
      
    def set_dataframe(self, dataframe):
        self.dataframe = dataframe
        
    def set_cur_scaler(self, scaler):
        self.cur_scaler = scaler

    def set_dataframe_PCA(self, dataframe):
        self.dataframe = dataframe
        self.standar_scaler = StandardScaler().fit_transform(dataframe)
        self.minmax_scaler = MinMaxScaler().fit_transform(dataframe)
        
    def set_varianza(self, varianza):
        self.varianza = varianza
    
    def set_predictor(self, predictor):
        self.predictor = predictor
    
    def get_preditor(self):
        return self.predictor
    
    def get_standar_scaler(self):
        return self.standar_scaler
    
    def get_minmax_scaler(self):
        return self.minmax_scaler
    
    def get_cur_scaler(self):
        return self.cur_scaler
    
    def get_df(self):
        return self.dataframe
    
    def get_varianza(self):
        return self.varianza
    
    def get_varianza_acum(self, value):
        if any(self.varianza != None):
            self.varianza_acumulada= sum(self.varianza[0:value])
            return self.varianza_acumulada
        return None