import pickle

'''This functions loads the model and predicts'''
def predict(data):
    model = pickle.load(open('dt.pkl','rb'))
    return model.predict(data)
print('Executed properly')