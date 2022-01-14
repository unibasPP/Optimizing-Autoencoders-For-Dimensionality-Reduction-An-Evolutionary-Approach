

# Model save and retriev
#-----------------------

## Save the model
#-----------------
import copy
import pickle


def save(data, name):
    # make deep copy of the model
    data1 = copy.deepcopy(data)
    # open a file in the binary-write mode and save model
    with open((name +".dat"), 'wb') as f:
        pickle.dump(data1, f)
    f.close()

def load(name):
    try:
        with open(name, 'rb') as f:
            data = pickle.load(f)
        f.close()
    except:
            data = 'Wrong directory or something else!'
    return data






