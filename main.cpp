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
    uniform_int_distribution<> dis(1, 100); 

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
        #pragma omp critical // Критична секція для оновлення глобальних змінних
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
    setlocale(LC_ALL, "en_US.UTF-8");
    
    omp_set_nested(1);
    
    int numThreads;
    cout << "Введіть кількість потоків: ";
    cin >> numThreads;
    omp_set_num_threads(numThreads);
    
    int rows, cols;
    cout << "Введіть кількість рядків: ";
    cin >> rows;
    cout << "Введіть кількість стовпців: ";
    cin >> cols;

    vector<vector<int>> matrix;
    fillMatrix(matrix, rows, cols);

    cout << "\nЗгенерована матриця:" << endl;
    printMatrix(matrix);

    int totalSum = 0;
    int minRowIdx = -1, minSum = INT_MAX;

    double t1, t2;
    t1 = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            totalSum = sumAllElements(matrix);
        }
        
        #pragma omp section
        {
            minRowSum(matrix, minRowIdx, minSum);
        }
    }

    t2 = omp_get_wtime();

    cout << "\nРезультати:" << endl;
    cout << "Загальна сума всіх елементів: " << totalSum << endl;
    cout << "Номер рядка з мінімальною сумою: " << minRowIdx + 1 << endl;
    cout << "Мінімальна сума елементів рядка: " << minSum << endl;
    cout << "Загальний час виконання (паралельне): " << (t2-t1) << " секунд" << endl;

    // Послідовне виконання
    int seqTotalSum = 0;
    int seqMinRowIdx = -1, seqMinSum = INT_MAX;

    double seq_t1, seq_t2;
    seq_t1 = omp_get_wtime();

    seqTotalSum = sumAllElementsSequential(matrix);
    minRowSumSequential(matrix, seqMinRowIdx, seqMinSum);

    seq_t2 = omp_get_wtime();

    cout << "\nРезультати послідовного виконання:" << endl;
    cout << "Загальна сума всіх елементів: " << seqTotalSum << endl;
    cout << "Номер рядка з мінімальною сумою: " << seqMinRowIdx + 1 << endl;
    cout << "Мінімальна сума елементів рядка: " << seqMinSum << endl;
    cout << "Загальний час виконання (послідовне): " << (seq_t2-seq_t1) << " секунд" << endl;


    cout << "\nНатисніть Enter для завершення...";
    cin.ignore();
    cin.get();
    return 0;
}