/*
 * ========================================
 * Dynamic Matrix Library - Implementation
 * ========================================
 * Implementation of the MatrixLib class methods
 */

#include "MatrixLib.h"

// ========================================
// CONSTRUCTORS
// ========================================
MatrixLib::MatrixLib(int r, int c) : rows(r), columns(c) {
    // Allocate memory for matrix data
    data = new float[rows * columns];
    
    // Initialize all elements to 0.0f
    for (int i = 0; i < rows * columns; i++) {
        data[i] = 0.0f;
    }
}

MatrixLib::MatrixLib(const MatrixLib& other) 
    : rows(other.rows), columns(other.columns) {
    // Allocate memory for the copy
    data = new float[rows * columns];

    // Copy all data elements
    for (int i = 0; i < rows * columns; i++) {
        data[i] = other.data[i];
    }
}

MatrixLib::MatrixLib(MatrixLib&& other) noexcept
    : rows(other.rows), columns(other.columns), data(other.data) {
    // Nullify the source matrix to prevent double deletion
    other.rows = 0;
    other.columns = 0;
    other.data = nullptr;
}

// ========================================
// DESTRUCTOR
// ========================================
MatrixLib::~MatrixLib() {
    delete[] data;
}

// ========================================
// ASSIGNMENT OPERATORS
// ========================================
MatrixLib& MatrixLib::operator=(MatrixLib other) {
    std::swap(rows, other.rows);
    std::swap(columns, other.columns);
    std::swap(data, other.data);
    return *this;
}

//MatrixLib& MatrixLib::operator=(MatrixLib&& other) noexcept {
//    if (this != &other) {
//        delete[] data;
//        rows = other.rows;
//        columns = other.columns;
//        data = other.data;
//        other.rows = 0;
//        other.columns = 0;
//        other.data = nullptr;
//    }
//    return *this;
// }

// ========================================
// ACCESSOR METHODS
// ========================================
float& MatrixLib::at(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= columns) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[r * columns + c];
}

const float& MatrixLib::at(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= columns) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[r * columns + c];
}

int MatrixLib::getRows() const {
    return rows;
}

int MatrixLib::getColumns() const {
    return columns;
}

// ========================================
// UTILITY METHODS
// ========================================
void MatrixLib::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(1) 
                      << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void MatrixLib::resize(int r, int c) {
    // Delete old data
    delete[] data;

    // Set new dimensions
    rows = r;
    columns = c;

    // Allocate new memory
    data = new float[rows * columns];

    // Initialize all elements to 0.0f
    for (int i = 0; i < rows * columns; i++) {
        data[i] = 0.0f;
    }
}

// ========================================
// ARITHMETIC OPERATIONS
// ========================================
MatrixLib MatrixLib::operator+(const MatrixLib& other) const {
    // Validate dimensions match
    if (rows != other.rows || columns != other.columns) {
        throw std::invalid_argument(
            "Cannot add matrices: dimensions must match"
        );
    }

    // Create result matrix with same dimensions
    MatrixLib result(rows, columns);

    // Add corresponding elements
    for (int i = 0; i < rows * columns; i++) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}



MatrixLib MatrixLib::operator-(const MatrixLib& other) const {
    if (rows != other.rows || columns != other.columns) {
        throw std::invalid_argument(
            "Cannot subtract matrices: dimensions must match"
        );
    }
    MatrixLib result(rows, columns);
    for (int i = 0; i < rows * columns; i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}
MatrixLib& MatrixLib::operator-=(const MatrixLib& other) {
    if (rows != other.rows || columns != other.columns) {
        throw std::invalid_argument(
            "Cannot subtract matrices: dimensions must match"
        );
    }
    for (int i = 0; i < rows * columns; i++) {
        data[i] -= other.data[i];
    }
    return *this;
}

MatrixLib& MatrixLib::operator+=(const MatrixLib& other) {
    if (rows != other.rows || columns != other.columns) {
        throw std::invalid_argument(
            "Cannot add matrices: dimensions must match"
        );
    }
    for (int i = 0; i < rows * columns; i++) {
        data[i] += other.data[i];
    }
    return *this;
}

MatrixLib MatrixLib::ScalarMultiply(float scalar) const {
    MatrixLib result(rows, columns);
    for (int i = 0; i < rows * columns; i++) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

MatrixLib MatrixLib::operator*(const MatrixLib& other) const {
    if (columns != other.rows) {
        throw std::invalid_argument(
            "Cannot multiply matrices: A's columns must match B's rows"
        );
    }
    
    MatrixLib result(rows, other.columns);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.columns; j++) {
            float sum = 0.0f;
            for (int k = 0; k < columns; k++) {
                sum += at(i, k) * other.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

MatrixLib MatrixLib::transpose() const {
    // Create result matrix with swapped dimensions (columns x rows)
    MatrixLib result(columns, rows);

    // Iterate through each element and swap row/column indices
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            // Element at [i][j] in original becomes [j][i] in transposed
            result.data[j * rows + i] = data[i * columns + j];
        }
    }

    return result;
}
