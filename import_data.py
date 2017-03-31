import numpy as np

def import_data(path):
    y, age, female, smoker, pc1, pc2, x1 = np.loadtxt(
        path,
        delimiter=',',
        dtype=[
            ('Y', '?'), # bool
            ('Age', 'u1'),
            ('Female', '?'),
            ('Smoker', '?'),
            ('PC1', 'f8'),
            ('PC2', 'f8'),
            ('X_1', '?'),
        ],
        skiprows=1,
        unpack=True,
    )

    return {
        'y': y,
        'age': age,
        'female': female,
        'smoker': smoker,
        'pc1': pc1,
        'pc2': pc2,
        'x1': x1
    }

# def shuffle(table, state=None):
#     table_copy = {k: v.copy() for k, v in table.items()}
#

if __name__ =='__main__':
    data = import_data('data/data.csv')
    print(data)
