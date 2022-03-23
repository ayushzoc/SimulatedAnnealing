from PIL import Image, ImageOps
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# import seaborn as sns

img = Image.open("test.png").convert("L")
WIDTH, HEIGHT = img.size

matrix = np.array(img)
matrix_2 = []
for i in range(matrix.shape[0]):
    if np.all(matrix[i] == 255):
        continue
    else:
        matrix_2.append(matrix[i])

matrix_2 = np.stack(matrix_2)
cities = matrix_2.T

citys = []
for i in range(cities.shape[0]):
    if np.all(cities[i] == 255):
        continue
    else:
        citys.append(cities[i])

citys = np.stack(cities)
CITY_COUNT = citys.shape[0]

final_image = Image.fromarray(citys.T, "L")
final_image.show()

distance_matrix = squareform(pdist(citys, "hamming"))
print(distance_matrix)

def Initialize(count):
    solution = np.arange(count)
    np.random.shuffle(solution)
    return solution

def Evaluate(cities, solution):
    distance = 0
    for i in range(cities.shape[0]-1):
        index_a = solution[i]
        index_b = solution[i + 1]
        distance += distance_matrix[index_a][index_b]
    return distance


def Modify(current):
    new = current.copy()
    x = np.random.randint(1, len(current) - 2)
    y = np.random.randint(1, len(current) - 2)
    while y == x or x == y + 1 or x == y - 1 or y == x + 1 or y == x - 1:
        y = np.random.randint(1, len(current) - 2)
    #     print(new[[x, y]])
    xplus, xminus = x + 1, x - 1
    yplus, yminus = y + 1, y - 1

    minus = distance_matrix[new[x]][new[xplus]] + distance_matrix[new[x]][new[xminus]] + distance_matrix[new[y]][
        new[yplus]] + distance_matrix[new[y]][new[yminus]]
    plus = distance_matrix[new[x]][new[yplus]] + distance_matrix[new[x]][new[yminus]] + distance_matrix[new[y]][
        new[xplus]] + distance_matrix[new[y]][new[xminus]]
    new[x], new[y] = new[y], new[x]

    return new, minus, plus

current_solution = Initialize(CITY_COUNT)
current_score = Evaluate(citys, current_solution)
P_start = 0.799
P_final = 10**-7
N = 80000000
# T_start = -1/np.log(P_start)
# T_start = 100
T_start = 50000
# T_final = -1/np.log(P_final)
T_final = 0.1
# F = (STOPPING_TEMPERATURE/INITIAL_TEMPERATURE)**(1/(N-1))
# F = (T_final/T_s/tart)**(1/(N-1))
F = 0.9999998
temperature = T_start
print(T_start)
print(T_final)
print("Current score: ", current_score)
print(current_solution.shape[0])
delta_avg_store = []

iteration = 1
while iteration <= N:
    new_solution, minus, plus = Modify(current_solution)
    new_score = current_score - minus + plus
    if plus < minus:
        current_solution = new_solution
        current_score = new_score
    else:
        probability = np.exp(-(plus-minus) / (temperature))
        if probability > np.random.uniform(0, 1):
            current_solution = new_solution
            current_score = new_score
    temperature *= F
    iteration += 1
print(Evaluate(citys, current_solution))

final_matrix = []
for index in current_solution:
    final_matrix.append(citys[index])
final_matrix = np.vstack(final_matrix)
print(final_matrix.T.shape)
final_image = Image.fromarray(final_matrix.T, "L")
final_image.save("image_final.jpg")
final_image.show()