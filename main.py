import pandas as pd
#import plotly.graph_objs as go
import numpy as np
import re, string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def mean_absolute_percentage_error(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Pre Processing of Texts
class PreProcessing():
    
  def __init__(self, min_df, max_df, ngram):
    self.min_df = min_df
    self.max_def = max_df
    self.ngram = ngram
    
  def CleanText(self, data):
    headlines = []

    for row in range(0, len(data)):
      text_top = str(data.iloc[row])      
      top_sub = re.sub('[0-9][^w]', '' , text_top)
      headlines.append(top_sub)
      
    return headlines
    
  def remove_stopwords(self, data, domain_stopwords=[]):
        
    stop_words = nltk.corpus.stopwords.words('english') # lang=portuguese or english
    dataheadlines = []
        
    for txt in data:
      s = str(txt).lower()
      table = str.maketrans({key: None for key in string.punctuation})
      s = s.translate(table)
      tokens = nltk.word_tokenize(s)
      estemas = [stemmer.stem(token) for token in tokens]
      v = [i for i in estemas if not i in stop_words and not i in domain_stopwords and not i.isdigit()] #remove stopwords     
      s = ""

      for token in v:
        s += token+" "
      
      dataheadlines.append(s.strip())
    
    return dataheadlines      
            
  def BoW(self, data):
    
    #data = self.GroupTexts(data, words)
    text = self.CleanText(data)
    headlines = self.remove_stopwords(text)
    #print(headlines)
             
    #TfIdfvectorizer = TfidfVectorizer(min_df=self.min_df, max_df= self.max_def, ngram_range = (self.ngram, self.ngram))
    TfIdfvectorizer = TfidfVectorizer(ngram_range = (1, 1), vocabulary=words)
    TrainingTFidf = TfIdfvectorizer.fit_transform(headlines)
    matrix_TFidf = pd.DataFrame(TrainingTFidf.todense(), columns=TfIdfvectorizer.get_feature_names())
    #print(TfIdfvectorizer.get_feature_names())

    return matrix_TFidf

  def GroupTexts(self, data):

    groupTexts = []
    
    for row in range(0, len(data)):
      news = str(data.iloc[row]).lower()      
      groupTexts.append(news)         

    return groupTexts

# ------------------------------------------------------------------------
# Time-series forecasting
class forecast_ts():

  def __init__(self):
    self.hdr = ['Win_len', 'MAPE', 'Value_True', 'Value_Pred']

  def prediction(self, base, model, x_st, y_st):
    
    df_ts = pd.DataFrame(columns=self.hdr)
    quant = 0
    mape = 0
    end = int(len(base) * 0.3)
    
    for x in range(end, len(base) - stps_fwd):
      
      win_ini = 0
      win_end = win_ini + x

      print("Training Window {} - {} - Test {}" .format(win_ini, win_end, win_end + 1 ))
      print("Days Remaining {}" .format( len(base) - (win_end + stps_fwd)))

      X_train = base.iloc[win_ini : win_end, x_st].values
      y_train = base.iloc[win_ini + stps_fwd: win_end + stps_fwd, y_st].values.ravel()
      X_test = base.iloc[win_end : win_end + 1, x_st].values
      y_test = base.iloc[win_end + stps_fwd : win_end + stps_fwd + stps_fwd , y_st].values.ravel()
     
      regression = model.fit(X_train, y_train)
      prediction = regression.predict(X_test)

      if (len(y_test) != stps_fwd):
        break

      print(y_test, " --- ", prediction)

      mape_pdt = mean_absolute_percentage_error(y_test[0], prediction)
      mape += mape_pdt
      quant += 1

      #print(y_test[0], prediction)
      
      df_ts = df_ts.append({'Win_len': win_end,
                            'MAPE': round(float(mape_pdt),2),
                            'Value_True': y_test[0], 
                            'Value_Pred': round(float(prediction),2)},
                            ignore_index=True)
      
    mape = round(float(mape / quant), 3)
    
    return df_ts, mape


def TS_Corn():

  res_ts_corn = pd.DataFrame(columns=hdr_best)
  pred1 = forecast_ts()
  
  x_st = [0, 1, 2, 3]
  y_st = [1]

  # CORN
  print("\n\tCORN PRICE FORECAST - TIMES SERIES\n")
  for key, model in Models.items():
    print("Calculando o Modelo: ", key)

    res_corn_ts_all, mape = pred1.prediction(base_corn, model, x_st, y_st)
    res_ts_corn = res_ts_corn.append({'Model': key,
                                      'Representation': 'ts_mul',
                                      'step': stps_fwd,
                                      'MAPE': mape,
                                      'True': list(res_corn_ts_all['Value_True']),
                                      'Pred': list(res_corn_ts_all['Value_Pred'])}, 
                                     ignore_index=True)
      
  return res_corn_ts_all, res_ts_corn

def TS_Soybean():

  res_ts_soy = pd.DataFrame(columns=hdr_best)
  pred2 = forecast_ts()
  
  x_st = [0, 1, 2, 3]
  y_st = [1]

  print("\n\t SOYBEAN PRICE FORECAST - TIMES SERIES\n")
  for key, model in Models.items():
    print("Processing the Model: ", key)
      
    res_soy_ts_all, mape = pred2.prediction(base_soybean, model, x_st, y_st)
    res_ts_soy = res_ts_soy.append({'Model': key,
                                    'Representation': 'ts_mul',
                                    'step': stps_fwd,
                                    'MAPE': mape,
                                    'True': list(res_soy_ts_all['Value_True']),
                                    'Pred': list(res_soy_ts_all['Value_Pred'])},
                                     ignore_index=True)
  
  return res_soy_ts_all, res_ts_soy

# ------------------------------------------------------------------------

# Forecasting for Time-Series Enriched with Domain-specific terms (TSED) and Domain-specific Terms (DST)
# If x_st = [0, 1, 2, 3], use attributes times series (TSED). 
# if x_st = [], not uses attributes times series (DST).

class forecasting():

  def __init__(self):
    self.hdr = ['Win_len', 'MAPE', 'Value_True', 'Value_Pred']

  def prediction(self, st_tx, model, min, max, gram, x_st, y_st):
    
    df_ts_bow = pd.DataFrame(columns=self.hdr)
    quant = 0
    mape = 0
    end = int(len(st_tx) * 0.3)
    
    for x in range(end, len(st_tx) - stps_fwd):
      
      win_ini = 0
      win_end = win_ini + x
      
      print("Training Window {} - {} - Test {}" .format(win_ini, win_end, win_end + 1 ))
      print("Days Remaining {}" .format( len(st_tx) - (win_end + stps_fwd)))
      
      text_wndw = st_tx.iloc[win_ini : win_end + 1, 5]
      PreProc = PreProcessing(min, max, gram)
      BoW_wndw = PreProc.BoW(text_wndw)
      #print(BoW_wndw)
      
      # Data extracted from the time series
      X_train_atb = st_tx.iloc[win_ini : win_end, x_st].values
      X_test_atb = st_tx.iloc[win_end : win_end + 1, x_st].values
      #print(X_test_atb.shape)

      # Atributes extracted from texts
      X_train_txt = BoW_wndw.iloc[win_ini : win_end, :]
      X_test_txt = BoW_wndw.iloc[win_end : win_end + 1, :]
      #print(X_test_txt)

      # Time series data and texts concatenated
      X_train = np.concatenate((X_train_atb, X_train_txt), axis=1)
      X_test = np.concatenate((X_test_atb, X_test_txt), axis=1)
      #print(X_test.shape)

      y_train = st_tx.iloc[win_ini + stps_fwd: win_end + stps_fwd, y_st].values.ravel()
      y_test = st_tx.iloc[win_end + stps_fwd : win_end + stps_fwd + stps_fwd , y_st].values.ravel()
      #print(y_train)
      #print(y_test)

      regression = model.fit(X_train, y_train)
      prediction = regression.predict(X_test)
      #print(y_test, " - ", prediction)

      #value = base_st.iloc[win_end - 1 : win_end, col].values + prediction
      if (len(y_test) != stps_fwd):
        break

      mape_pdt = mean_absolute_percentage_error(y_test[0], prediction)
      mape += mape_pdt
      quant += 1
    
      df_ts_bow = df_ts_bow.append({'Win_len': win_end,
                                    'MAPE': round(float(mape_pdt),2),
                                    'Value_True': y_test[0], 
                                    'Value_Pred': round(float(prediction),2)},
                                   ignore_index=True)

    mape = round(float(mape / quant), 3)
    
    return df_ts_bow, mape


def TS_ED_Corn():

  res_ts_corn = pd.DataFrame(columns=hdr_best)
  pred3 = forecasting()
  
  x_st = [0, 1, 2, 3]
  y_st = [1]

  # CORN
  print("\tCORN PRICE FORECAST - TS_ED\n")
  for key, model in Models.items():
    
    print("Processing the Model: ", key)
      
    res_corn_BoW_all, mape = pred3.prediction(corn_st_tx, model, 0, 1, 1, x_st, y_st) #dataset, model, 0%, 1%, unigram
    res_ts_corn = res_ts_corn.append({'Model': key,
                                      'Representation': 'tsed',
                                      'step': stps_fwd,
                                      'MAPE': mape,
                                      'True': list(res_corn_BoW_all['Value_True']),
                                      'Pred': list(res_corn_BoW_all['Value_Pred'])},
                                     ignore_index=True)
      
      #print("Janela de Treino: ", x, ' - MAPE:', mape)
 
  return res_corn_BoW_all, res_ts_corn

def TS_ED_Soybean():

  # SOYBEAN
  res_ts_soy = pd.DataFrame(columns=hdr_best)
  pred4 = forecasting()
  
  x_st = [0, 1, 2, 3]
  y_st = [1]

  print("\n\t SOYBEAN PRICE FORECAST - TS_ED")
  for key, model in Models.items():

    print("Processing the Model: ", key)

    res_soy_BoW_all, mape = pred4.prediction(soy_st_tx, model, 0, 1, 1, x_st, y_st)
    res_ts_soy = res_ts_soy.append({'Model': key,
                                    'Representation': 'tsed',
                                    'step': stps_fwd,
                                    'MAPE': mape,
                                    'True': list(res_soy_BoW_all['Value_True']),
                                    'Pred': list(res_soy_BoW_all['Value_Pred'])},
                                   ignore_index=True)

  return res_soy_BoW_all, res_ts_soy


# ------------------------------------------------------------------------

# Forecasting with Domain Specific Terms only (DST)

def DST_Corn():

  res_ts_corn = pd.DataFrame(columns=hdr_best)
  pred5 = forecasting()
  
  x_st = []
  y_st = [1]

  # CORN
  print("\tCORN PRICE FORECAST - DST\n")
  for key, model in Models.items():
    
    print("Processing the Model: ", key)
      
    res_corn_BoW_all, mape = pred5.prediction(corn_st_tx, model, 0, 1, 1, x_st, y_st) #dataset, model, 0%, 1%, unigram
    res_ts_corn = res_ts_corn.append({'Model': key,
                                      'Representation': 'dst',
                                      'step': stps_fwd,
                                      'MAPE': mape,
                                      'True': list(res_corn_BoW_all['Value_True']),
                                      'Pred': list(res_corn_BoW_all['Value_Pred'])},
                                     ignore_index=True)
      
      #print("Janela de Treino: ", x, ' - MAPE:', mape)
 
  return res_corn_BoW_all, res_ts_corn

def DST_Soybean():

  # SOYBEAN
  res_ts_soy = pd.DataFrame(columns=hdr_best)
  pred6 = forecasting()
  
  x_st = []
  y_st = [1]

  print("\n\t SOYBEAN PRICE FORECAST - DST")
  for key, model in Models.items():

    print("Processing the Model: ", key)

    res_soy_BoW_all, mape = pred6.prediction(soy_st_tx, model, 0, 1, 1, x_st, y_st)
    res_ts_soy = res_ts_soy.append({'Model': key, 
                                    'MAPE': mape,
                                    'Representation': 'dst',
                                    'step': stps_fwd,
                                    'True': list(res_soy_BoW_all['Value_True']),
                                    'Pred': list(res_soy_BoW_all['Value_Pred'])},
                                   ignore_index=True)

  return res_soy_BoW_all, res_ts_soy

# ------------------------------------------------------------------------

def main():
    
    global hdr, hdr_best, Models, win, words, batch_len, layers_len, epoch_len, corn_st_tx, soy_st_tx, stemmer
    hdr = ['Diary', 'Values_True', 'Values_Pred']
    hdr_best = ['Model', 'Representation', 'step', 'MAPE', 'True', 'Pred']

    global base_corn    
    base_corn = pd.read_csv('corn.csv', header=0, delimiter=",")
    base_corn['Date'] = pd.to_datetime(base_corn['Date'])
    base_corn.dropna(subset=['Close', 'Open', 'High', 'Low'], inplace=True)
    base_corn.drop(columns=['Volume'], axis=1, inplace=True)
    base_corn.sort_values("Date", ascending=True, inplace=True)
    
    global base_soybean
    base_soybean = pd.read_csv('soybean.csv', header=0, delimiter=",")
    base_soybean['Date'] = pd.to_datetime(base_soybean['Date'])
    base_soybean.dropna(subset=['Close', 'Open', 'High', 'Low'], inplace=True)
    base_soybean.drop(columns=['Volume'], axis=1, inplace=True)
    base_soybean.sort_values("Date", ascending=True, inplace=True)
    
    Models = {"HGradBoostingRreg": HistGradientBoostingRegressor(),
              "SVR_RBF": SVR(kernel='rbf', gamma='auto'),
              "RandomForrest4": RandomForestRegressor(max_depth=4, random_state=0),
              "BaggingReg": BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)}
        
    
    stemmer = SnowballStemmer('english')
    
    KeyWords = ["crop", "safrinha", "losses", "yield", "estimate", "disappoint", "excellent", "good", "rains", "planting", 
               "increase", " decrease", "price", "reduction", "sales", "additional", "complete", "lower", "low", "more", "progress",
               "high", "domestic", "harvest", "production", "decline", "cost", "export", "import", "no news", "record", "large", "growing"]
    
    words = set(stemmer.stem(d) for d in KeyWords)
    
    #read text files
    texts = pd.read_excel('news.xlsx')
    texts['Date'] = pd.to_datetime(texts['Date'])
    texts.drop(columns='Headlines', axis=1, inplace=True)
    texts.dropna(inplace=True)
    texts.sort_values("Date", ascending=True, inplace=True)
    texts.drop_duplicates(subset="Date", keep="first", inplace=True)
    
    #Days when there was no news on the site, the term “no news” was considered for training and testing
    #concatenate ST and Texts - Corn. 
    corn_st_tx = pd.merge(base_corn, texts, on='Date', how='outer')
    corn_st_tx['News'].replace({np.nan: "no news"}, inplace=True)
    #print(corn_st_tx)
    
    #concatenate ST and Texts - Soybean. 
    soy_st_tx = pd.merge(base_soybean, texts, on='Date', how='outer')
    soy_st_tx['News'].replace({np.nan: "no news"}, inplace=True)
    #print(soy_st_tx)
    
    global stps_fwd
    # Forecasting 1-5 steps forward - Corn
    for t in range(1,6):
        
        # Steps forward
        stps_fwd = t
        print("\n\nProcessing ", t, " steps forward\n\n")
        
        df_mape_corn = pd.DataFrame(columns=hdr_best)
        
        #Forecasting with TS
        _, res_ts_corn = TS_Corn()
        df_mape_corn = df_mape_corn.append(res_ts_corn, ignore_index=True)
        
        #Forecasting with TSED
        _, res_tsed_corn = TS_ED_Corn()
        df_mape_corn = df_mape_corn.append(res_tsed_corn, ignore_index=True)
        
        #Forcasting with DST
        _, res_dst_corn = DST_Corn()
        df_mape_corn = df_mape_corn.append(res_dst_corn, ignore_index=True)
        
    df_mape_corn.to_csv("corn_res1.csv", sep='\t', index=False)
    
    # Forecasting 1-5 steps forward - Soybean
    for t in range(1,6):
        
        # Steps forward
        stps_fwd = t
        print("\n\nProcessing ", t, " steps forward\n\n")
        
        df_mape_soy = pd.DataFrame(columns=hdr_best)
        
        #Forecasting with TS
        _, res_ts_soy = TS_Soybean()
        df_mape_soy = df_mape_soy.append(res_ts_soy, ignore_index=True)
        
        #Forecasting with TSED
        _, res_tsed_soy = TS_ED_Soybean()
        df_mape_soy = df_mape_soy.append(res_tsed_soy, ignore_index=True)
        
        #Forecasting with DST
        _, res_dst_soy = DST_Soybean()
        df_mape_soy = df_mape_soy.append(res_dst_soy, ignore_index=True) 
    
    df_mape_soy.to_csv("soybean_res1.csv", sep='\t', index=False)
    
    
if __name__ == '__main__':
	main()

