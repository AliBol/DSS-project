import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(10)
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
import random
random.seed(10)
import torch
torch.manual_seed(10)
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import RobertaModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import scipy.stats
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler


split_seed = 3
name_final = '../checkpoints/roberta_movies_3_final_add.pt'
name_best = '../checkpoints/roberta_movies_3_best_add.pt'

df_cellphone = pd.read_csv('../dataset/movies.csv.gz', compression='gzip')

new_df = df_cellphone["asin"].unique()

df_cellphone = pd.read_csv('../dataset/LIWC-22 Results - movies_complete - LIWC Analysis.csv')

rating_twod = df_cellphone['overall'].values.reshape(-1,1)
print(rating_twod)
numerical_transformer = StandardScaler()
star_rating_norm = numerical_transformer.fit_transform(rating_twod)
df_cellphone['normalized_rating'] = star_rating_norm



wc_twod = df_cellphone['WC'].values.reshape(-1,1)
print(wc_twod)
numerical_transformer = StandardScaler()
wc_norm = numerical_transformer.fit_transform(wc_twod)
df_cellphone['normalized_wc'] = wc_norm



print(len(new_df))


training_samp, test_samp = train_test_split(new_df, train_size=0.8, test_size=0.2, random_state= split_seed)
training_samp, val_samp = train_test_split(training_samp, test_size=0.125, random_state= split_seed)
print(len(training_samp))
print(len(test_samp))
print(len(val_samp))
training_data = df_cellphone[df_cellphone['asin'].isin(training_samp)]
test_data = df_cellphone[df_cellphone['asin'].isin(test_samp)]
val_data = df_cellphone[df_cellphone['asin'].isin(val_samp)]
print(len(training_data))
print(len(test_data))
print(len(val_data))


df_subsample_train = training_data[['reviewText','helpfulness_ratio','asin','normalized_rating','normalized_wc']].sample(frac = 1, random_state = split_seed)
df_subsample_test = test_data[['reviewText','helpfulness_ratio','asin','normalized_rating','normalized_wc']].sample(frac = 1, random_state = split_seed)
df_subsample_val = val_data[['reviewText','helpfulness_ratio','asin','normalized_rating','normalized_wc']].sample(frac = 1, random_state = split_seed)

X_train = list(df_subsample_train['reviewText'])
y_train = list(df_subsample_train['helpfulness_ratio'])
X_test = list(df_subsample_test['reviewText'])
y_test = list(df_subsample_test['helpfulness_ratio'])
X_val = list(df_subsample_val['reviewText'])
y_val = list(df_subsample_val['helpfulness_ratio'])

X_train_add = list(df_subsample_train[['normalized_rating','normalized_wc']].values)
X_val_add = list(df_subsample_val[['normalized_rating','normalized_wc']].values)
X_test_add = list(df_subsample_test[['normalized_rating','normalized_wc']].values)

print("len X_train_add=", len(X_train_add))

bias_linear = np.mean(y_train)
print("bias_linear=", bias_linear)

tokenizer_2 = RobertaTokenizerFast.from_pretrained("roberta-base")


train_encodings = tokenizer_2(X_train, padding = 'max_length', truncation = 'longest_first', max_length=300, return_attention_mask=True)
test_encodings = tokenizer_2(X_test, padding = 'max_length', truncation = 'longest_first', max_length =300, return_attention_mask=True)
val_encodings = tokenizer_2(X_val, padding = 'max_length', truncation = 'longest_first', max_length = 300, return_attention_mask=True)


train_masks = train_encodings['attention_mask']
test_masks = test_encodings['attention_mask']
train_inputs = train_encodings['input_ids']
test_inputs = test_encodings['input_ids']
val_inputs = val_encodings['input_ids']
val_masks = val_encodings['attention_mask']




batch_size = 16
def create_dataloaders(inputs, inputs_add, masks, labels, batch_size,shuffle=False):
    input_tensor = torch.tensor(inputs)
    input_add_tensor = torch.tensor(inputs_add).float()
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, input_add_tensor, mask_tensor, 
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle)
    return dataloader

train_dataloader = create_dataloaders(train_inputs, X_train_add, train_masks, 
                                      y_train, batch_size,shuffle=True)
test_dataloader = create_dataloaders(test_inputs, X_test_add, test_masks, 
                                     y_test, batch_size)
val_dataloader = create_dataloaders(val_inputs, X_val_add, val_masks, 
                                      y_val, batch_size)




@torch.no_grad()
def init_bias(m):
    if type(m) == nn.Linear:
        m.bias.fill_(bias_linear)

class RobertaRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.2, freeze_Roberta=False, freeze_layers='all', add_dim=0):
        
        super(RobertaRegressor, self).__init__()
        D_in, D_out = 768+add_dim, 1
        D_hid = int(D_in/2)

        self.freeze_Roberta = freeze_Roberta
        
        self.roberta = \
                   RobertaModel.from_pretrained('roberta-base')
        if self.freeze_Roberta:
          if freeze_layers=='all':
              for param in self.roberta.parameters():
                  param.requires_grad = False
          else:
              for layer in self.roberta.encoder.layer[:freeze_layers]:
                  for param in layer.parameters():
                      param.requires_grad = False
        self.cat_layer = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_hid),nn.ReLU())        
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_hid, D_out))
        self.regressor.apply(init_bias)

        
    def forward(self, input_ids, attention_masks, input_add):
        
        outputs = self.roberta(input_ids, attention_masks)
        class_label_output = outputs[1]
        

        
        hidden = self.cat_layer(torch.cat((class_label_output,input_add),dim=1))
        outputs = self.regressor(hidden)
        return outputs
model = RobertaRegressor(drop_rate=0.0, freeze_Roberta=True, freeze_layers=10, add_dim=2)


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=1e-4,
                  weight_decay = 0.0,
                  eps=1e-8)

epochs = 5
total_steps = len(train_dataloader) * epochs * 10
warmup_steps = len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer,       
                 num_warmup_steps=warmup_steps, num_training_steps=total_steps)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to saved checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer ussed in training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize best_loss from checkpoint to best_loss
    best_loss = checkpoint['best_loss']
    return model, optimizer, checkpoint['epoch'], best_loss

loss_function = nn.MSELoss()

def evaluate(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss= []
    for batch in test_dataloader:
        batch_inputs, batch_inputs_add, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks, batch_inputs_add)
        loss = loss_function(outputs.squeeze(), batch_labels.squeeze())
        test_loss.append(loss.item())
    return np.mean(test_loss)


#This is just to format the time that training takes
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


#To save some stats of training
training_stats = []
#starting time of training
t_total_start = time.time()
from torch.nn.utils.clip_grad import clip_grad_norm
def train(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, device, save_name_best, save_name_final, clip_value=2):
    best_loss = 1e10
    ev = evaluate(model, loss_function, val_dataloader, device) 
    best_loss = ev
    for epoch in range(epochs):
        print(epoch)
        #time of this epoch
        t_e_start = time.time()
        
        model.train()
        #To keep sum of loss over training batches
        running_loss = 0.
        for step, batch in enumerate(train_dataloader): 
            if step % 250 == 0:
                print(step, 'time elapsed for epoch',format_time(time.time() - t_e_start))  
            batch_inputs, batch_inputs_add, batch_masks, batch_labels = \
                               tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks, batch_inputs_add)           
            loss = loss_function(outputs.squeeze(), 
                             batch_labels.squeeze())
            loss.backward()
            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        training_time_epoch = format_time(time.time() - t_e_start)
        print('training loss', running_loss/len(train_dataloader), 'epoch training time', training_time_epoch)
        ev = evaluate(model, loss_function, val_dataloader, device) 
        print('validation metrics: ', ev)
        print("-----")
        if ev < best_loss:
          best_loss = ev
          torch.save({'epoch':epoch,'best_loss':best_loss,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}, save_name_best)
        training_stats.append(
        {
            'epoch': epoch,
            'training loss': running_loss/len(train_dataloader),
            'valid loss': ev,
            'training time': training_time_epoch
        }
        )
    
    print('time elapsed for training',format_time(time.time() - t_total_start))     
    torch.save({'epoch':epoch,'best_loss':best_loss,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}, save_name_final)            
    return model
model = train(model, optimizer, scheduler, loss_function, epochs, 
              train_dataloader, device, name_best, name_final, clip_value=1000)

#create a pandas df from training stats
df_stats = pd.DataFrame(data=training_stats)

df_stats = df_stats.set_index('epoch')

# Display the table.
print(df_stats)



# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['training loss'], 'b-o', label="Training")
plt.plot(df_stats['valid loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks(list(range(epochs)))

plt.savefig('../figs/roberta_movies_3_add.png', dpi = 300)

def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_inputs_add, batch_masks, _ = \
                                  tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, 
                            batch_masks, batch_inputs_add).view(1,-1).tolist()[0]
    return output

model, optimizer, best_epoch, best_loss = load_ckp(name_best, model, optimizer)
y_pred = predict(model, test_dataloader, device)
y_pred_val = predict(model, val_dataloader, device)



mae = mean_absolute_error(y_test, y_pred)
mdae = median_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


pearson = scipy.stats.pearsonr(y_test, y_pred)
spearman = scipy.stats.spearmanr(y_test, y_pred)
kendall = scipy.stats.kendalltau(y_test, y_pred)

mae_val = mean_absolute_error(y_val, y_pred_val)
mdae_val = median_absolute_error(y_val, y_pred_val)
mse_val = mean_squared_error(y_val, y_pred_val)


pearson_val = scipy.stats.pearsonr(y_val, y_pred_val)
spearman_val = scipy.stats.spearmanr(y_val, y_pred_val)
kendall_val = scipy.stats.kendalltau(y_val, y_pred_val)

df_subsample_test['prediction'] = np.squeeze(y_pred)
df_subsample_val['prediction'] = np.squeeze(y_pred_val)

print("MAE =" , mae)
print("MDAE =" ,mdae)
print("MSE =" ,mse)
print("RMSE =" ,np.sqrt(mse))
print("PPC =" ,pearson)
print("SPC =" ,spearman)
print("KC =" ,kendall)

print('validation_metrics')

print("MAE =" , mae_val)
print("MDAE =" ,mdae_val)
print("MSE =" ,mse_val)
print("RMSE =" ,np.sqrt(mse_val))
print("PPC =" ,pearson_val)
print("SPC =" ,spearman_val)
print("KC =" ,kendall_val)

df_subsample_test['number_review_product'] = df_subsample_test.groupby('asin')['helpfulness_ratio'].transform('count')
df_subsample_test['mean_helpfulness_product'] = df_subsample_test.groupby('asin')['helpfulness_ratio'].transform('mean')

df_subsample_test_ndcg=df_subsample_test[(df_subsample_test['number_review_product']>1) &(df_subsample_test['mean_helpfulness_product']>0)].groupby('asin')\
.apply(lambda x: ndcg_score(x['helpfulness_ratio'].to_numpy().reshape((1,-1)),x['prediction'].to_numpy().reshape((1,-1)),k=10))

print("test_ndcg =", df_subsample_test_ndcg.mean())
print('shape_ndcg = ', df_subsample_test.shape)

print(df_subsample_test[(df_subsample_test['number_review_product']>1) &(df_subsample_test['mean_helpfulness_product']>0)].shape)

df_subsample_val['number_review_product'] = df_subsample_val.groupby('asin')['helpfulness_ratio'].transform('count')
df_subsample_val['mean_helpfulness_product'] = df_subsample_val.groupby('asin')['helpfulness_ratio'].transform('mean')
df_subsample_val_ndcg=df_subsample_val[(df_subsample_val['number_review_product']>1) &(df_subsample_val['mean_helpfulness_product']>0)].groupby('asin')\
.apply(lambda x: ndcg_score(x['helpfulness_ratio'].to_numpy().reshape((1,-1)),x['prediction'].to_numpy().reshape((1,-1)),k=10))
print('val_ndcg_mean =', df_subsample_val_ndcg.mean())