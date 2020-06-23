#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Importando bibliotecas
import numpy as np
import pandas as pd


# In[20]:


from sklearn.preprocessing import OneHotEncoder


# # Definindo a classe

# In[21]:


class database:
    
    def __init__(self,pasta = ''):
        
        
        self.calendar = pd.read_csv(pasta+'calendar.csv')
        self.sales_train_evaluation = pd.read_csv(pasta+'sales_train_evaluation.csv')
        self.sales_train_validation = pd.read_csv(pasta+'sales_train_validation.csv')
        self.sample_submission = pd.read_csv(pasta+'sample_submission.csv')
        self.sell_prices = pd.read_csv(pasta+'sell_prices.csv')
        
        # Pre-processamento do calendario
        self.calendar = self.calendar.drop(labels = ['event_name_1','event_name_2','date'], axis = 1)
        values = {'event_type_1': 'No Event','event_type_2': 'No Event'}
        self.calendar = self.calendar.fillna(value = values)
        
        # Variáveis categóricas
        self.OHT = OneHotEncoder(sparse = False)     
        cat_data = self.calendar.iloc[:,list(pd.Series(self.calendar.columns).isin(['event_type_1','event_type_2','weekday','month']))]
        
        self.OHT.fit(cat_data)
        
        cat_data = self.OHT.transform(cat_data)
        cat_data = pd.DataFrame(cat_data)
        
        labels = np.concatenate(self.OHT.categories_)
        cat_data.columns = labels
        
        # Juntando calendario e váriaveis categoricas
        self.calendar = self.calendar.drop(labels = ['event_type_1','event_type_2','weekday','month'], axis = 1)
        self.calendar = pd.concat([self.calendar,cat_data],axis=1)
        
        
        
        self.item_id =  self.sales_train_validation['item_id'].value_counts().index.sort_values()
        self.dept_id =  self.sales_train_validation['dept_id'].value_counts().index.sort_values()
        self.cat_id =  self.sales_train_validation['cat_id'].value_counts().index.sort_values()
        self.store_id = self.sales_train_validation['store_id'].value_counts().index.sort_values()
        self.state_id = self.sales_train_validation['state_id'].value_counts().index.sort_values()
        
        self.id =  self.sales_train_validation['id'].value_counts().index.sort_values()
        
        
        
    def get_labels(self, dataset = 'evaluation'):
        
        if dataset == 'evaluation':
            labels = self.sales_train_evaluation
            sales_id = self.sales_train_evaluation['id']
        elif dataset == 'validation':
            labels = self.sales_train_validation
            sales_id = self.sales_train_validation['id']
        labels = pd.DataFrame(labels.iloc[:,6:])
        labels = labels.transpose()
        labels.columns = sales_id
        
        self.eval_labels = labels
        
        return labels

    def get_item_features(self,item,days):
        item = item.split('_')
        
        item_id = item[0] + '_' + item[1]+ '_' + item[2]
        store_id = item[3] + '_' + item[4]
        
        sell_prices = self.sell_prices
        calendar = self.calendar
        
        sell_prices_update = sell_prices[sell_prices['store_id'] == store_id ][sell_prices['item_id'] == item_id ]
        calendar_update = calendar.iloc[:days,:]
        calendar_update = calendar_update[calendar_update['wm_yr_wk'].isin(sell_prices_update['wm_yr_wk'])]
        
        
        sell_prices_final = sell_prices_update
        
        for i in range(6):
            sell_prices_final = pd.concat([sell_prices_final, sell_prices_update])
            
        sell_prices_final = sell_prices_final.sort_values('wm_yr_wk')
        
        calendar_update = calendar_update.reset_index(drop = True)

        spf_size = len(sell_prices_final)
        cu_size = len(calendar_update)
        final_week_diff = 7 - calendar_update['wday'][cu_size-1]
        first_week_diff = calendar_update['wday'][0]-1
        
        
        sell_prices_final = sell_prices_final.iloc[first_week_diff:spf_size -final_week_diff,:]
        
        
        calendar_update['sell_price'] = sell_prices_final['sell_price'].reset_index(drop = True)
        
        return calendar_update
    
    
    
    # item full id example 'FOODS_1_001_CA_1_validation'
    def Hierarchy(self, cat='NaN', dept = 'NaN', store = 'NaN', state = 'NaN' ):
        
        hierarchycal_itens_list = []
        aux = 0
        
        
        item_ref = [cat, dept,"NaN", state, store,'NaN']
        
        for i in self.id:
            
            item = i.split('_')
            item[2] = 'NaN'
            item[5] = 'NaN'
            
            if cat == 'NaN':
                item[0] = 'NaN'
            if dept == 'NaN':
                item[1] = 'NaN'
            if state == 'NaN':
                item[3] = 'NaN'
            if store == 'NaN':
                item[4] = 'NaN'         
            
            if item_ref == item:
                hierarchycal_itens_list.append(i)
                
        return hierarchycal_itens_list


# ## Testando dados

# In[22]:


pasta = 'Dados-originais/'
M5db = database(pasta = pasta)


# In[23]:


M5db.calendar.head()


# In[ ]:


M5db.sales_train_evaluation.head()


# In[ ]:


M5db.sales_train_validation.head()


# In[ ]:


M5db.sell_prices.head()


# In[ ]:


M5db.item_id


# In[ ]:


M5db.dept_id


# In[ ]:


M5db.cat_id


# In[ ]:


M5db.store_id


# In[ ]:


M5db.state_id


# # Testando Funções

# ## database.get_labels( dataset)
# Retorna as labels do dataset desejado

# In[ ]:


M5db.get_labels(dataset = 'validation')


# In[ ]:


M5db.get_labels(dataset = 'evaluation')


# ## database.get_item_features( Item, Loja, Data )
# Retorna todas features de um item de uma determinada loja até a data especificada

# In[ ]:


get_ipython().run_cell_magic('time', '', "M5db.get_item_features('FOODS_1_003_TX_1_validation',1913)")


# ## database.Hierarchy( cat, dept, store, state )
# Retorna todos ids de itens em uma ramificação da hierarquia especificada

# In[ ]:


get_ipython().run_cell_magic('time', '', "M5db.Hierarchy(cat='HOUSEHOLD', dept = '2', store = '2', state = 'TX' )")


# In[ ]:


#Pegando todos itens de todas categorias só do estado do texas
M5db.Hierarchy(state = 'TX' )

