Проект представляет собой реализацию метода BiCGSTAB (Biconjugate Gradient Stabilized Method) для решения разреженных линейных систем, с поддержкой распараллеливания операций.

Авторы: Силаев Дмитрий (гр. 23162), Стучинский Арсений (гр. 23161), Малышев Сергей (гр. 23162)

## О методе BiCGSTAB

**BiCGSTAB (Biconjugate Gradient Stabilized)** — это итеративный метод для решения больших разреженных систем линейных уравнений вида $Ax = b$, где:
- $A$ — матрица системы,
- $b$ — правая часть,
- $x$ — вектор неизвестных.

Этот метод улучшает классический метод биконъюгированных градиентов (BiCG) за счет стабилизирующих шагов, которые обеспечивают лучшую сходимость.

### Идея метода

Метод BiCGSTAB основан на поиске приближенного решения $x$ с использованием следующих идей:

1. **Метод остаточных значений**:
   - Вычисляется остаток:
     $r_k = b - Ax_k,$
     который измеряет, насколько текущее приближение $x_k$ близко к истинному решению.

2. **Поиск в подпространствах**:
   - Используются два направления для корректировки приближения:
     - $p_k$: направление из метода BiCG.
     - $s_k$: вспомогательное направление для стабилизации.

3. **Обновление решения**:
   - Решение $x_k$ обновляется в каждой итерации с использованием коррекции, основанной на:
     - Скалярных произведениях.
     - Операциях с остатками.
   - Формула обновления:
     $x_{k+1} = x_k + \alpha p_k + \omega s_k,$
     где $\alpha$ и $\omega$ — коэффициенты, определяющие вклад каждого направления.


### Алгоритм BiCGSTAB

Основные шаги алгоритма:

1. **Инициализация**:
   - Задать начальное приближение $x_0$.
   - Вычислить начальный остаток:
     $r_0 = b - Ax_0$
   - Выбрать вспомогательный вектор $r_0$ для ортогонализации.

2. **Итерации**:
   - На каждой итерации вычисляются:
     - Коэффициенты $\rho, \alpha, \beta, \omega$, которые используются для обновления направлений и стабилизации сходимости.
   - Построить направления:
     $p_k \quad \text{и} \quad s_k$
   - Обновить текущее решение:
     $x_{k+1} = x_k + \alpha p_k + \omega s_k$
   - Вычислить новый остаток:
     $r_{k+1} = b - Ax_{k+1}$

3. **Проверка сходимости**:
   - Если выполнено условие:
     $\|r_k\| \leq \text{tol},$
     где $\text{tol}$ — допустимая ошибка, метод завершается.

### Объяснение коэффициентов
- $\rho$ — корректирует направление $p_k$.
- $\alpha$ — масштабирует компонент $p_k$.
- $\beta$ — учитывает предыдущее направление для улучшения текущего.
- $\omega$ — стабилизирует сходимость, компенсируя колебания.

Метод повторяет итерации до выполнения условия сходимости или достижения максимального количества итераций.

### Распределение ролей для каждого файла в проекте:

---

### **Файл `bicgstab.h`**
Этот заголовочный файл:
- Содержит объявление класса **`BiCGSTAB`**, который реализует метод BiCGSTAB.
- Предоставляет интерфейс для выполнения алгоритма через метод:
  - `run`: выполняет метод BiCGSTAB с заданными параметрами (матрица, вектор, точность, количество итераций).
- Поддерживает выбор между последовательным и параллельным выполнением алгоритма (через флаг `usingParallel`).

---

### **Файл `bicgstab.cpp`**
Этот файл реализует:
1. **Метод BiCGSTAB**:
   - `bicgstab_parallel`: версия метода с использованием OpenMP для параллельного выполнения.
   - `bicgstab`: последовательная версия метода.
   - Оба метода используют формат CSR (Compressed Sparse Row) для работы с разреженными матрицами.
2. **Утилиты для вычислений**:
   - `sparseMatrixVectorParallelMultiplyCSR`: параллельное умножение матрицы CSR на вектор.
   - `sparseMatrixVectorMultiplyCSR`: последовательное умножение матрицы CSR на вектор.
   - `parallelDotProduct`: параллельное скалярное произведение векторов.
   - `dotProduct`: последовательное скалярное произведение.
3. **Метод `run`**:
   - Объединяет последовательную и параллельную версии алгоритма, в зависимости от флага `usingParallel`.

---

### **Файл `tests.cpp`**
Этот файл выполняет:
1. **Тестирование BiCGSTAB**:
   - Тесты сравнивают производительность и точность между последовательной (`bicgstab`) и параллельной (`bicgstab_parallel`) версиями.
   - Используются заранее подготовленные матрицы в формате CSR.
2. **Генерация данных**:
   - `generateSparseMatrixCSR`: создает разреженные матрицы случайного размера в формате CSR.
3. **Сравнение с библиотекой Eigen**:
   - Конвертирует CSR-матрицы в формат `Eigen::SparseMatrix`.
   - Использует BiCGSTAB из библиотеки Eigen для проверки корректности и производительности.
4. **Функции тестирования**:
   - `test_with_params`: выполняет тестирование на данных, заданных параметрами.
   - `test_with_generate_matrix`: генерирует случайные матрицы и запускает тесты.
   - Отдельные тесты:
     - `test1`, `test1_1`, `test2`, `test2_1`: тесты на корректность.
     - `test3`, `test3_not_parallel`: тесты для больших матриц с использованием многопоточности.
5. **Главная функция (`main`)**:
   - Запускает тесты на корректность, сравнение производительности между последовательным и параллельным подходами, а также сравнение с библиотекой Eigen.

---

### **Файл `graphic.ipynb`**
Этот Jupyter Notebook выполняет анализ результатов вычислений и визуализацию:
1. **Данные**:
   - Включает набор данных, содержащий значения времени вычислений для различных конфигураций (с количеством потоков 1, 2, 4 и использование Eigen).
2. **Предобработка данных**:
   - Данные сгруппированы по размеру матрицы, а затем усредняются каждые три строки, чтобы уменьшить шум.
3. **Построение графиков**:
   - Создаёт график зависимости времени вычислений от размера матрицы для каждой из конфигураций.
   - Используется библиотека `matplotlib` для визуализации, а также `pandas` и `numpy` для обработки данных.

---

### **Краткое назначение файлов**
| **Файл**          | **Роль**                                                                 |
|--------------------|--------------------------------------------------------------------------|
| `bicgstab.h`      | Объявление класса `BiCGSTAB` и интерфейса для выполнения алгоритма.       |
| `bicgstab.cpp`    | Реализация метода BiCGSTAB и вспомогательных функций.                    |
| `tests.cpp`       | Тестирование метода, генерация данных, сравнение с Eigen.                |
| `graphic.ipynb`   | Анализ времени вычислений и визуализация.                |

Эти файлы вместе обеспечивают полный цикл: от реализации метода BiCGSTAB до тестирования его производительности и корректности.

![image](https://github.com/user-attachments/assets/6be8d2ec-2d85-4486-b5dc-2471c9dea8b3)





