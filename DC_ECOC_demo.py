"""
There is an example of using OVO_ECOC class to validate on dermatology data
"""
from ECOCDemo.ECOC.Classifier import DC_ECOC
from read import read_UCI_Dataset

filepath = r'E:\workspace\pycharm\UCI\dermatology.csv'
data, label = read_UCI_Dataset(filepath)

#new ECOC model
E = DC_ECOC()
E.fit(data, label)

true_label = label[0]
predicted_label = E.predict(data[0])

#calculate the 3K cross validation accuracy
accuracy = E.validate(data, label)

# print accuracy
print("accuracy:", accuracy)
