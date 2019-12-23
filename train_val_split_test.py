from sklearn.model_selection import train_test_split
import numpy as np

x=np.arange(0,10,1)

train_x,test_x=train_test_split(x,test_size=0.2)

print(train_x)

print(test_x)




