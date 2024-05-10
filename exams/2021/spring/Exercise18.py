import numpy as np

template = [208, 233, 71,
            231, 161, 139,
            32, 25, 244]

t = np.sum(template)

image = [167, 193, 180,
         9, 189, 8,
         217, 100, 71]

print(np.sum(image)/t)


import numpy as np

im = np.array([[167, 193, 180],
               [9, 189, 8],
               [217, 100, 71]])

tem = np.array([[208, 233, 71],
                [231, 161, 139],
                [32, 25, 244]])

im = im / np.linalg.norm(im)
tem = tem / np.linalg.norm(tem)

cosine_similarity = np.sum(im * tem) / np.sqrt(np.sum(im**2) * np.sum(tem**2))
print(cosine_similarity)