from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

global features

def output_results(model, data):
    y_pred = model.predict(data[features])
    print("Prediction Report\n", classification_report(data['lossOrNot'], y_pred))
    cm = confusion_matrix(data['lossOrNot'], y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(data=cm, linewidths=.5, annot=True, square = True, cmap = 'Blues', fmt='d')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# download the data in csv format from https://mplanestore.polito.it:5001/sharing/wp4uP94x1
data = pd.read_csv('df_losses_features_target.csv')

# our dataset includes some addtional info, which should not be included into testing 
features = []
for col in data.columns:
    
    '''
    timestamp represents the time bin
    lossOrNot is the class label (0: no-loss, 1: loss)
    flow_id is the numerical ID assigned to each specific flow
    num_packet_loss indicates the number of packet loss in a bin, which will be used to create reduced dataset
    '''
    if col not in ['timestamp', 'lossOrNot', 'flow_id', 'num_packet_loss']:
        
        # we include all the features in the past 10s,
        # but due to the time cosntraint, we need to discard those that are in the past time bin
        if '_500ms' not in col:
            features.append(col)

# xgboost for the entire dataset
model_xgb_1 = xgb.sklearn.XGBClassifier()
model_xgb_1.load_model("xgb_model_entire_dataset.txt")
output_results(model_xgb_1, data)

# brf for the reduced dataset
with open('brf_model_reduced_dataset.pkl', 'rb') as f:
    model_brf = pickle.load(f)
output_results(model_brf, data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)])
      
# xgb for the reduced dataset
model_xgb_2 = xgb.sklearn.XGBClassifier()
model_xgb_2.load_model("xgb_model_reduced_dataset.txt")
output_results(model_xgb_2, data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)])