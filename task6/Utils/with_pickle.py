import pickle


def save_to(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_from(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data
