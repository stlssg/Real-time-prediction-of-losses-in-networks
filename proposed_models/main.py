from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('df_losses_features_target.csv')

features = []
for col in data.columns:
    if col not in ['timestamp', 'lossOrNot', 'flow_id', 'num_packet_loss']:
        if '_500ms' not in col:
            features.append(col)

# xgboost for the entire dataset
model_xgb_1 = xgb.sklearn.XGBClassifier()
model_xgb_1.load_model("xgb_model_entire_dataset.txt")

y_pred = model_xgb_1.predict(data[features])
print("Prediction Report\n", classification_report(data['lossOrNot'], y_pred))
cm = confusion_matrix(data['lossOrNot'], y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm, linewidths=.5, annot=True, square = True, cmap = 'Blues', fmt='d')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# brf for the reduced dataset
with open('brf_model_reduced_dataset.pkl', 'rb') as f:
    model = pickle.load(f)
      
y_pred = model.predict(data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)][features])
print("Prediction Report\n", classification_report(data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)]['lossOrNot'], y_pred))
cm = confusion_matrix(data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)]['lossOrNot'], y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm, linewidths=.5, annot=True, square = True, cmap = 'Blues', fmt='d')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# xgb for the reduced dataset
model_xgb_2 = xgb.sklearn.XGBClassifier()
model_xgb_2.load_model("xgb_model_reduced_dataset.txt")
      
y_pred = model_xgb_2.predict(data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)][features])
print("Prediction Report\n", classification_report(data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)]['lossOrNot'], y_pred))
cm = confusion_matrix(data[(data['num_packet_loss']>2) | (data['num_packet_loss']==0)]['lossOrNot'], y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm, linewidths=.5, annot=True, square = True, cmap = 'Blues', fmt='d')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()