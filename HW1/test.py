import numpy as np

# Создаем лист листов
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Преобразуем в numpy массив
arr = np.array(nested_list)

# Находим индексы числа 5
indices = np.where(arr == 5)

# Выводим результат
print("Индексы числа 5:", indices)

import numpy as np

# Создаем матрицу с разным количеством элементов в каждой строке
nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

# Преобразуем в numpy массив
arr = np.array([np.array(row) for row in nested_list])

# Находим индексы числа 5
target_number = 5
indices = np.where(np.concatenate([row == target_number for row in nested_list]))

# Выводим результат
print("Индексы числа", target_number, "в несимметричной матрице:", indices)
