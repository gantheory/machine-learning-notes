import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

target_vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
source_vectors = [[4, 5, 6], [7, 8, 9], [1, 2, 3]]

similarity = cosine_similarity(target_vectors, source_vectors)

result = np.argmax(similarity, axis=0)

print(similarity)
print(result)

"""
output:
    [[ 0.97463185  0.95941195  1.        ]
    [ 1.          0.99819089  0.97463185]
    [ 0.99819089  1.          0.95941195]]
    [1 2 0]
"""
