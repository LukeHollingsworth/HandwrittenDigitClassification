from scipy import io as sio
import numpy as np


def load_data():
    mat = sio.loadmat(r'C:\Users\Luke Hollingsworth\Documents\Other\Python Scripts\SamAssisstant\data\matlab\emnist-letters.mat')
    data = mat['dataset']

    training_images = data['train'][0,0]['images'][0,0]
    training_labels = data['train'][0,0]['labels'][0,0]
    test_images = data['test'][0,0]['images'][0,0]
    test_labels = data['test'][0,0]['labels'][0,0]

    validation_start = training_images.shape[0] - test_images.shape[0]
    validation_images = training_images[validation_start:training_images.shape[0],:]
    validation_labels = training_labels[validation_start:training_images.shape[0]]

    training_images = training_images[0:validation_start, :]
    training_labels = training_labels[0:validation_start]
    
    return (training_images, training_labels, validation_images, validation_labels, test_images, test_labels)

def load_data_wrapper():
    tr_i, tr_l, va_i, va_l, te_i, te_l = load_data()
    training_inputs = [np.reshape(np.array(tr_i), (784,1))]
    training_results = [vectorized_result(y) for y in tr_l[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(np.array(va_i), (784,1))]
    validation_data = zip(validation_inputs, va_l[1])
    test_inputs = [np.reshape(np.array(te_i), (784,1))]
    test_data = zip(test_inputs, te_l[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((26, 1))
    e[j] = 1.0
    return e