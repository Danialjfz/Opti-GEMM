/*
 * ========================================
 * Dynamic Matrix Library - Header File
 * ========================================
 * A C++ library for creating and manipulating matrices
 * Features: Matrix operations, arithmetic, and transformations
 */

#pragma once
#ifndef MATRIXLIB_H
#define MATRIXLIB_H
#include <iostream>
#include <stdexcept>
#include <utility>
#include <iomanip>
#include <climits>

extern volatile float sink;

// ========================================
// MATRIX CLASS DEFINITION
// ========================================
class MatrixLib {
    private:
        // ========================================
        // PRIVATE MEMBER VARIABLES
        // ========================================
        int rows;           // Number of rows
        int columns;        // Number of columns
        float* data;        // Dynamic array to store matrix elements (row-major order)

    public:
        // ========================================
        // CONSTRUCTORS
        // ========================================
        /*
         * MatrixLib(int r, int c)
         * Initializes a matrix with specified dimensions
         * All elements are initialized to 0.0f
         * Parameters:
         *   r - number of rows
         *   c - number of columns
         */
        MatrixLib(int r, int c);

        /*
         * MatrixLib(const MatrixLib& other)
         * Copy Constructor - Creates a deep copy of another matrix
         * Parameters:
         *   other - the matrix to copy
         */
        MatrixLib(const MatrixLib& other);

        /*
         * MatrixLib(MatrixLib&& other) noexcept
         * Move Constructor - Efficiently transfers resources from temporary
         * Parameters:
         *   other - the temporary matrix to move
         */
        MatrixLib(MatrixLib&& other) noexcept;

        // ========================================
        // DESTRUCTOR
        // ========================================
        /*
         * ~MatrixLib()
         * Deallocates memory used by the matrix
         * Called when matrix object goes out of scope
         */
        ~MatrixLib();

        // ========================================
        // ASSIGNMENT OPERATORS
        // ========================================
        /*
         * MatrixLib& operator=(MatrixLib other)
         * Assignment Operator using copy-and-swap idiom
         * Provides exception safety and automatic cleanup
         * Parameters:
         *   other - the matrix to assign
         * Returns: reference to this matrix
         */
        MatrixLib& operator=(MatrixLib other);

        /*
         * MatrixLib& operator=(MatrixLib&& other) noexcept
         * Move Assignment Operator - Efficiently transfers resources from temporary
         * Parameters:
         *   other - the temporary matrix to move
         * Returns: reference to this matrix
         */
        //MatrixLib& operator=(MatrixLib&& other) noexcept;

        // ========================================
        // ACCESSOR METHODS
        // ========================================
        /*
         * float& at(int r, int c)
         * Returns a reference to the element at position (r, c)
         * Allows both reading and writing to matrix elements
         * Includes bounds checking for safety
         * Parameters:
         *   r - row index
         *   c - column index
         * Returns: reference to the element at (r, c)
         * Throws: std::out_of_range if indices are invalid
         */
        float& at(int r, int c);

        /*
         * const float& at(int r, int c) const
         * Const version of at() for read-only access
         * Parameters:
         *   r - row index
         *   c - column index
         * Returns: const reference to the element at (r, c)
         * Throws: std::out_of_range if indices are invalid
         */
        const float& at(int r, int c) const;

        /*
         * int getRows() const
         * Returns the number of rows in the matrix
         */
        int getRows() const;

        /*
         * int getColumns() const
         * Returns the number of columns in the matrix
         */
        int getColumns() const;

        // ========================================
        // UTILITY METHODS
        // ========================================
        /*
         * void print() const
         * Prints the matrix in a formatted grid layout
         * Each row is printed on a new line with elements separated by spaces
         */
        void print() const;

        /*
         * void resize(int r, int c)
         * Changes the dimensions of the matrix
         * Deallocates old memory and allocates new memory
         * All previous data is lost and new elements are initialized to 0.0f
         * Parameters:
         *   r - new number of rows
         *   c - new number of columns
         */
        void resize(int r, int c);

        // ========================================
        // ARITHMETIC OPERATIONS
        // ========================================
        /*
         * MatrixLib add(const MatrixLib& other) const
         * Adds two matrices element-wise
         * Both matrices must have the same dimensions
         * Parameters:
         *   other - the matrix to add to this matrix
         * Returns: a new matrix containing the sum
         * Throws: std::invalid_argument if dimensions don't match
         */
     
        MatrixLib operator+(const MatrixLib& other) const;

        /*
         * MatrixLib operator-(const MatrixLib& other) const
         * Subtracts one matrix from another element-wise
         * Both matrices must have the same dimensions
         * Parameters:
         *   other - the matrix to subtract
         * Returns: a new matrix containing the difference
         * Throws: std::invalid_argument if dimensions don't match
         */
        MatrixLib operator-(const MatrixLib& other) const;

        /*
         * MatrixLib operator+=(const MatrixLib& other)
         * Compound assignment operator for addition
         * Adds another matrix to this matrix in-place
         * Parameters:
         *   other - the matrix to add
         * Returns: reference to this matrix
         * Throws: std::invalid_argument if dimensions don't match
         */
        MatrixLib& operator+=(const MatrixLib& other);

        /*
         * MatrixLib& operator-=(const MatrixLib& other)
         * Compound assignment operator for subtraction
         * Subtracts another matrix from this matrix in-place
         * Parameters:
         *   other - the matrix to subtract
         * Returns: reference to this matrix
         * Throws: std::invalid_argument if dimensions don't match
         */
        MatrixLib& operator-=(const MatrixLib& other);

        /*
         * MatrixLib ScalarMultiply(float scalar) const
         * Multiplies each element of the matrix by a scalar value
         * Parameters:
         *   scalar - the value to multiply each element by
         * Returns: a new matrix with scaled elements
         */
        MatrixLib ScalarMultiply(float scalar) const;

        /*
         * MatrixLib operator*(const MatrixLib& other) const
         * Performs matrix multiplication (dot product)
         * This matrix's columns must equal other matrix's rows
         * Result dimensions: (this.rows x other.columns)
         * Parameters:
         *   other - the matrix to multiply with
         * Returns: a new matrix containing the product
         * Throws: std::invalid_argument if dimensions are incompatible
         */
        MatrixLib operator*(const MatrixLib& other) const;

        /*
         * MatrixLib transpose() const
         * Computes the transpose of the matrix (rows become columns, columns become rows)
         * Returns: a new matrix that is the transpose of this matrix
         * Dimensions of result: (this.columns x this.rows)
         */
        MatrixLib transpose() const;
};

#endif
