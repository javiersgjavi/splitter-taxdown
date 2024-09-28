import numpy as np

def compare_strings(str1, str2):
    print(f'str1:\n {str1}\n')
    print(f'str2:\n {str2}')
    

def average_pairwise_cosine_similarity(list1, list2):
    # Convertir las listas a arrays numpy
    array1 = np.array(list1)
    array2 = np.array(list2)
    
    # Calcular la matriz de similitud coseno entre todos los pares, se hace con el dot product, como recomienda la documentacion de openai
    similarity_matrix = np.dot(array1, array2.T)
    
    # Calcular el promedio de todas las similitudes
    average_similarity = np.mean(similarity_matrix)
    
    return average_similarity