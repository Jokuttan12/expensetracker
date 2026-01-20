import numpy as np
import pandas as pd
import joblib as jb
import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

data=yf.download('AAPL',start='2015-01-01',auto_adjust=True)["Close"].values.reshape(-1,1)
print(f"data shape:{data.shape}")
print(f"firest three:{data[:3]}")
print(f"last three:{data[-3:]}")


scaler=MinMaxScaler(feature_range=(0,1))
data_scaled=scaler.fit_transform(data)
jb.dump(scaler,'scaler.pk1')


print(f"scaled range:{data_scaled.min():.3f}->{data_scaled.max():.3f}")

print("creating sequences")
def create_sequences(data,seq_length=60):
    X,Y=[],[]
    for i in range(seq_length,len(data)):
        X.append(data[i-seq_length:i,0])
        Y.append(data[i,0])
    return np.array(X),np.array(Y)

seq_length=60
X,Y =create_sequences(data_scaled)
X=X.reshape((X.shape[0],X.shape[1],1))

print(f"X shape:{X.shape},Y shape:{Y.shape}")
print(X)

split=int(0.8 *len(X))
x_train,x_test=X[:split],X[split:]
y_train,y_test=Y[:split],Y[split:]

np.save('x_test.npy',x_test)
np.save('y_test.npy',y_test)

print(f"train:{x_train.shape},test:{x_test.shape}")


model=Sequential([
    LSTM(50,return_sequences=True,
         input_shape=(seq_length,1)),
         Dropout(0.2),
         LSTM(50,return_sequences=False),
         Dropout(0.2),
         Dense(25),
         Dense(1)
])

model.compile(optimizer='adam',
              loss='mean_squared_error')
model.summary()



history=model.fit(x_train,y_train,
          batch_size=32,
          epochs=30,
          validation_data=(x_test,y_test),
          verbose=1)
model.save('lstm_model.h5')


plt.figure(figsize=(10, 6)) 
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_test, label='Actual Prices', alpha=0.8)
plt.plot(y_pred, label='LSTM Predictions', alpha=0.8)
plt.title('AAPL: Actual vs Predicted (Test Set)')
plt.ylabel('Price ($)')
plt.xlabel('Test Days')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('predictions.png')
plt.show()

# ===== 5. DIRECTIONAL ACCURACY =====
actual_direction = np.diff(y_test_real) > 0
pred_direction = np.diff(y_pred_real) > 0
directional_acc = np.mean(actual_direction == pred_direction) * 100
print(f"Directional Accuracy: {directional_acc:.1f}% (up/down correct)")
print("="*50)