#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>
#include <locale>
#include <clocale>
#include <climits>

using namespace std;

void fillMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    // Ініціалізація генератора випадкових чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100); // Генерація чисел від 1 до 100

    matrix.resize(rows);
    for (int i = 0; i < rows; ++i) {
        matrix[i].resize(cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
}

void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            cout << val << "\t";
        }
        cout << endl;
    }
}

// Функція для знаходження суми всіх елементів матриці
// Використовує OpenMP для паралельного обчислення
int sumAllElements(const vector<vector<int>>& matrix) {
    int totalSum = 0;
    #pragma omp parallel for reduction(+:totalSum) // Паралельне обчислення з редукцією
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            totalSum += matrix[i][j];
        }
    }
    return totalSum;
}

// Функція для знаходження рядка з мінімальною сумою елементів
// Використовує OpenMP для паралельного обчислення
void minRowSum(const vector<vector<int>>& matrix, int& minRowIdx, int& minSum) {
    minSum = INT_MAX;
    minRowIdx = -1;
    #pragma omp parallel
    {
        int localMinSum = INT_MAX;
        int localMinRowIdx = -1;
        #pragma omp for
        for (int i = 0; i < matrix.size(); ++i) {
            int rowSum = 0;
            for (int j = 0; j < matrix[i].size(); ++j) {
                rowSum += matrix[i][j];
            }
            if (rowSum < localMinSum) {
                localMinSum = rowSum;
                localMinRowIdx = i;
            }
        }
        #pragma omp critical // Критична секція для безпечного оновлення глобальних змінних
        {
            if (localMinSum < minSum) {
                minSum = localMinSum;
                minRowIdx = localMinRowIdx;
            }
        }
    }
}

// Послідовна версія функції для знаходження суми всіх елементів матриці
int sumAllElementsSequential(const vector<vector<int>>& matrix) {
    int totalSum = 0;
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            totalSum += matrix[i][j];
        }
    }
    return totalSum;
}

// Послідовна версія функції для знаходження рядка з мінімальною сумою елементів
void minRowSumSequential(const vector<vector<int>>& matrix, int& minRowIdx, int& minSum) {
    minSum = INT_MAX;
    minRowIdx = -1;
    for (int i = 0; i < matrix.size(); ++i) {
        int rowSum = 0;
        for (int j = 0; j < matrix[i].size(); ++j) {
            rowSum += matrix[i][j];
        }
        if (rowSum < minSum) {
            minSum = rowSum;
            minRowIdx = i;
        }
    }
}

int main() {
    // Налаштування локалі для коректного відображення кирилиці
    setlocale(LC_ALL, "en_US.UTF-8");
    
    // Введення кількості потоків
    int numThreads;
    cout << "Введіть кількість потоків: ";
    cin >> numThreads;
    omp_set_num_threads(numThreads);
    
    // Введення розмірів матриці
    int rows, cols;
    cout << "Введіть кількість рядків: ";
    cin >> rows;
    cout << "Введіть кількість стовпців: ";
    cin >> cols;

    // Створення та заповнення матриці
    vector<vector<int>> matrix;
    fillMatrix(matrix, rows, cols);

    // Виведення згенерованої матриці
    cout << "\nЗгенерована матриця:" << endl;
    printMatrix(matrix);

    // Змінні для зберігання результатів
    int totalSum = 0, totalSumSeq = 0;
    int minRowIdx = -1, minSum = 0;
    int minRowIdxSeq = -1, minSumSeq = 0;

    // Змінні для вимірювання часу виконання
    double t1, t2, t3, t4, t5, t6, t7, t8;

    // Послідовне виконання
    t1 = omp_get_wtime();
    totalSumSeq = sumAllElementsSequential(matrix);
    t2 = omp_get_wtime();
    
    t3 = omp_get_wtime();
    minRowSumSequential(matrix, minRowIdxSeq, minSumSeq);
    t4 = omp_get_wtime();

    // Паралельне виконання
    t5 = omp_get_wtime();
    totalSum = sumAllElements(matrix);
    t6 = omp_get_wtime();
    
    t7 = omp_get_wtime();
    minRowSum(matrix, minRowIdx, minSum);
    t8 = omp_get_wtime();

    // Виведення результатів
    cout << "\nРезультати:" << endl;
    cout << "Загальна сума всіх елементів (послідовно): " << totalSumSeq << endl;
    cout << "Загальна сума всіх елементів (паралельно): " << totalSum << endl;
    cout << "Час обчислення суми (послідовно): " << (t2-t1) << " секунд" << endl;
    cout << "Час обчислення суми (паралельно): " << (t6-t5) << " секунд" << endl;
    
    cout << "\nНомер рядка з мінімальною сумою (послідовно): " << minRowIdxSeq + 1 << endl;
    cout << "Номер рядка з мінімальною сумою (паралельно): " << minRowIdx + 1 << endl;
    cout << "Мінімальна сума елементів рядка (послідовно): " << minSumSeq << endl;
    cout << "Мінімальна сума елементів рядка (паралельно): " << minSum << endl;
    cout << "Час пошуку мінімального рядка (послідовно): " << (t4-t3) << " секунд" << endl;
    cout << "Час пошуку мінімального рядка (паралельно): " << (t8-t7) << " секунд" << endl;

    // Очікування натискання Enter перед завершенням програми
    cout << "\nНатисніть Enter для завершення...";
    cin.ignore();
    cin.get();
    return 0;
}