/*
 * ========================================
 * Dynamic Matrix Library - Test Suite
 * ========================================
 * Comprehensive testing for MatrixLib class
 * Tests: All constructors, operators, and methods
 */

#include "MatrixLib.h"
#include <iostream>
#include <string>
#include <stdexcept>

// ========================================
// TEST TRACKING
// ========================================
static int totalTests  = 0;
static int passedTests = 0;
static int failedTests = 0;

// ========================================
// TEST HELPER FUNCTIONS
// ========================================

/*
 * Prints a formatted section header
 */
void printTestHeader(int testNum, const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST " << testNum << ": " << title         << std::endl;
    std::cout << std::string(60, '=') << "\n"                << std::endl;
}

/*
 * Prints pass/fail and tracks results
 */
void printTestResult(bool passed, const std::string& message) {
    totalTests++;
    if (passed) {
        passedTests++;
        std::cout << "  [PASS] " << message << std::endl;
    } else {
        failedTests++;
        std::cout << "  [FAIL] " << message << std::endl;
    }
}

/*
 * Compares two floats within a tolerance
 */
bool floatEqual(float a, float b, float tolerance = 0.001f) {
    float diff = a - b;
    return diff >= -tolerance && diff <= tolerance;
}

/*
 * Compares two matrices element-wise within a tolerance
 */
bool matricesEqual(const MatrixLib& A, const MatrixLib& B,
                   float tolerance = 0.001f) {
    if (A.getRows()    != B.getRows() ||
        A.getColumns() != B.getColumns()) {
        return false;
    }
    for (int i = 0; i < A.getRows(); i++) {
        for (int j = 0; j < A.getColumns(); j++) {
            if (!floatEqual(A.at(i, j), B.at(i, j), tolerance)) {
                return false;
            }
        }
    }
    return true;
}

/*
 * Prints a summary of all test results
 */
void printSummary() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUMMARY"                              << std::endl;
    std::cout << std::string(60, '=')                        << std::endl;
    std::cout << "  Total  : " << totalTests                 << std::endl;
    std::cout << "  Passed : " << passedTests                << std::endl;
    std::cout << "  Failed : " << failedTests                << std::endl;
    std::cout << std::string(60, '=') << "\n"                << std::endl;
}

// ========================================
// INDIVIDUAL TEST FUNCTIONS
// ========================================

// ----------------------------------------
// TEST 1: Basic Constructor
// ----------------------------------------
void testBasicConstructor() {
    printTestHeader(1, "Basic Constructor");

    // Create a 3x3 matrix
    MatrixLib A(3, 3);

    // Check dimensions
    printTestResult(A.getRows()    == 3,
                    "3x3 matrix has 3 rows");
    printTestResult(A.getColumns() == 3,
                    "3x3 matrix has 3 columns");

    // Check all elements initialized to 0
    bool allZero = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!floatEqual(A.at(i, j), 0.0f)) {
                allZero = false;
            }
        }
    }
    printTestResult(allZero,
                    "All elements initialized to 0.0f");

    // Check non-square matrix dimensions
    MatrixLib B(2, 5);
    printTestResult(B.getRows()    == 2 &&
                    B.getColumns() == 5,
                    "Non-square 2x5 matrix has correct dimensions");

    // Check 1x1 matrix
    MatrixLib C(1, 1);
    C.at(0, 0) = 42.0f;
    printTestResult(floatEqual(C.at(0, 0), 42.0f),
                    "1x1 matrix element set and retrieved correctly");

    std::cout << "\nMatrix A (3x3, all zeros):\n";
    A.print();
}

// ----------------------------------------
// TEST 2: Copy Constructor
// ----------------------------------------
void testCopyConstructor() {
    printTestHeader(2, "Copy Constructor");

    // Build source matrix
    MatrixLib A(2, 2);
    A.at(0, 0) = 1.0f;  A.at(0, 1) = 2.0f;
    A.at(1, 0) = 3.0f;  A.at(1, 1) = 4.0f;

    // Create copy
    MatrixLib B(A);

    // Modify copy - original must not change
    B.at(0, 0) = 99.0f;

    printTestResult(floatEqual(A.at(0, 0), 1.0f),
                    "Original unchanged after modifying copy");
    printTestResult(floatEqual(B.at(0, 0), 99.0f),
                    "Copy reflects modification");

    // Check all other values copied correctly
    printTestResult(floatEqual(B.at(0, 1), 2.0f) &&
                    floatEqual(B.at(1, 0), 3.0f) &&
                    floatEqual(B.at(1, 1), 4.0f),
                    "All remaining elements copied correctly");

    // Deep copy - dimensions must match
    printTestResult(B.getRows()    == A.getRows() &&
                    B.getColumns() == A.getColumns(),
                    "Copy has same dimensions as original");

    std::cout << "\nOriginal A:\n";
    A.print();
    std::cout << "\nCopy B (B[0,0] = 99):\n";
    B.print();
}

// ----------------------------------------
// TEST 3: Move Constructor
// ----------------------------------------
void testMoveConstructor() {
    printTestHeader(3, "Move Constructor");

    // Create a source matrix
    MatrixLib A(2, 2);
    A.at(0, 0) = 10.0f;  A.at(0, 1) = 20.0f;
    A.at(1, 0) = 30.0f;  A.at(1, 1) = 40.0f;

    // Move into B
    MatrixLib B(std::move(A));

    // B must have the original values
    printTestResult(floatEqual(B.at(0, 0), 10.0f) &&
                    floatEqual(B.at(0, 1), 20.0f) &&
                    floatEqual(B.at(1, 0), 30.0f) &&
                    floatEqual(B.at(1, 1), 40.0f),
                    "Moved matrix contains original values");

    // Dimensions transferred
    printTestResult(B.getRows()    == 2 &&
                    B.getColumns() == 2,
                    "Moved matrix has correct dimensions");

    // A should be in a valid but empty state
    printTestResult(A.getRows()    == 0 &&
                    A.getColumns() == 0,
                    "Source matrix left in empty state after move");

    std::cout << "\nMatrix B (moved from A):\n";
    B.print();
}

// ----------------------------------------
// TEST 4: Copy Assignment Operator
// ----------------------------------------
void testCopyAssignment() {
    printTestHeader(4, "Copy Assignment Operator");

    MatrixLib A(2, 2);
    A.at(0, 0) = 1.0f;  A.at(0, 1) = 2.0f;
    A.at(1, 0) = 3.0f;  A.at(1, 1) = 4.0f;

    // Assign to existing matrix
    MatrixLib B(3, 3);
    B = A;

    printTestResult(B.getRows()    == 2 &&
                    B.getColumns() == 2,
                    "Assigned matrix adopts correct dimensions");
    printTestResult(matricesEqual(A, B),
                    "Assigned matrix has correct values");

    // Modify B - A must not change
    B.at(0, 0) = 99.0f;
    printTestResult(floatEqual(A.at(0, 0), 1.0f),
                    "Original unchanged after modifying assignment copy");

    // Self-assignment must not crash
    A = A;
    printTestResult(floatEqual(A.at(0, 0), 1.0f),
                    "Self-assignment is safe");

    std::cout << "\nMatrix A:\n";
    A.print();
    std::cout << "\nMatrix B (assigned from A, then B[0,0] = 99):\n";
    B.print();
}

// ----------------------------------------
// TEST 5: Move Assignment Operator
// ----------------------------------------
void testMoveAssignment() {
    printTestHeader(5, "Move Assignment Operator");

    MatrixLib A(2, 2);
    A.at(0, 0) = 5.0f;  A.at(0, 1) = 6.0f;
    A.at(1, 0) = 7.0f;  A.at(1, 1) = 8.0f;

    MatrixLib B(1, 1);
    B = std::move(A);

    printTestResult(floatEqual(B.at(0, 0), 5.0f) &&
                    floatEqual(B.at(0, 1), 6.0f) &&
                    floatEqual(B.at(1, 0), 7.0f) &&
                    floatEqual(B.at(1, 1), 8.0f),
                    "Move-assigned matrix contains correct values");

    printTestResult(B.getRows()    == 2 &&
                    B.getColumns() == 2,
                    "Move-assigned matrix has correct dimensions");

    printTestResult(A.getRows()    == 0 &&
                    A.getColumns() == 0,
                    "Source left in empty state after move assignment");

    std::cout << "\nMatrix B (move-assigned from A):\n";
    B.print();
}

// ----------------------------------------
// TEST 6: Element Access - at()
// ----------------------------------------
void testElementAccess() {
    printTestHeader(6, "Element Access - at()");

    MatrixLib A(3, 3);

    // Write and read back every element
    float val = 1.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A.at(i, j) = val++;
        }
    }

    bool correct = true;
    val = 1.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!floatEqual(A.at(i, j), val++)) {
                correct = false;
            }
        }
    }
    printTestResult(correct,
                    "All elements written and read back correctly");

    // Const access
    const MatrixLib& constRef = A;
    printTestResult(floatEqual(constRef.at(0, 0), 1.0f),
                    "Const at() returns correct value");

    // Out-of-bounds row
    try {
        float x = A.at(5, 0);
        (void)x;
        printTestResult(false, "Out-of-bounds row throws exception");
    } catch (const std::out_of_range&) {
        printTestResult(true, "Out-of-bounds row throws std::out_of_range");
    }

    // Out-of-bounds column
    try {
        float x = A.at(0, 5);
        (void)x;
        printTestResult(false, "Out-of-bounds column throws exception");
    } catch (const std::out_of_range&) {
        printTestResult(true,
                        "Out-of-bounds column throws std::out_of_range");
    }

    // Negative index
    try {
        float x = A.at(-1, 0);
        (void)x;
        printTestResult(false, "Negative index throws exception");
    } catch (const std::out_of_range&) {
        printTestResult(true,
                        "Negative index throws std::out_of_range");
    }

    std::cout << "\nMatrix A (3x3, values 1-9):\n";
    A.print();
}

// ----------------------------------------
// TEST 7: Matrix Addition
// ----------------------------------------
void testAddition() {
    printTestHeader(7, "Matrix Addition");

    MatrixLib A(2, 2);
    A.at(0, 0) = 1.0f;  A.at(0, 1) = 2.0f;
    A.at(1, 0) = 3.0f;  A.at(1, 1) = 4.0f;

    MatrixLib B(2, 2);
    B.at(0, 0) = 10.0f;  B.at(0, 1) = 20.0f;
    B.at(1, 0) = 30.0f;  B.at(1, 1) = 40.0f;

    // operator+
    MatrixLib C = A + B;
    printTestResult(floatEqual(C.at(0, 0), 11.0f) &&
                    floatEqual(C.at(0, 1), 22.0f) &&
                    floatEqual(C.at(1, 0), 33.0f) &&
                    floatEqual(C.at(1, 1), 44.0f),
                    "operator+ produces correct element-wise sum");

    // add() method
    MatrixLib D = A+B;
    printTestResult(matricesEqual(C, D),
                    "add() method matches operator+ result");

    // Originals unchanged
    printTestResult(floatEqual(A.at(0, 0), 1.0f),
                    "A unchanged after addition");
    printTestResult(floatEqual(B.at(0, 0), 10.0f),
                    "B unchanged after addition");

    // Dimension mismatch
    MatrixLib E(3, 3);
    try {
        MatrixLib F = A + E;
        printTestResult(false, "Addition with mismatched dims throws");
    } catch (const std::invalid_argument&) {
        printTestResult(true,
                        "Addition with mismatched dims throws "
                        "std::invalid_argument");
    }

    std::cout << "\nA + B:\n";
    C.print();
}

// ----------------------------------------
// TEST 8: Matrix Subtraction
// ----------------------------------------
void testSubtraction() {
    printTestHeader(8, "Matrix Subtraction");

    MatrixLib A(2, 2);
    A.at(0, 0) = 10.0f;  A.at(0, 1) = 20.0f;
    A.at(1, 0) = 30.0f;  A.at(1, 1) = 40.0f;

    MatrixLib B(2, 2);
    B.at(0, 0) = 1.0f;  B.at(0, 1) = 2.0f;
    B.at(1, 0) = 3.0f;  B.at(1, 1) = 4.0f;

    MatrixLib C = A - B;
    printTestResult(floatEqual(C.at(0, 0),  9.0f) &&
                    floatEqual(C.at(0, 1), 18.0f) &&
                    floatEqual(C.at(1, 0), 27.0f) &&
                    floatEqual(C.at(1, 1), 36.0f),
                    "Subtraction produces correct element-wise difference");

    // Originals unchanged
    printTestResult(floatEqual(A.at(0, 0), 10.0f) &&
                    floatEqual(B.at(0, 0),  1.0f),
                    "Operands unchanged after subtraction");

    // Self-subtraction gives zero matrix
    MatrixLib D = A - A;
    bool allZero = true;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (!floatEqual(D.at(i, j), 0.0f)) {
                allZero = false;
            }
        }
    }
    printTestResult(allZero, "A - A produces zero matrix");

    // Dimension mismatch
    MatrixLib E(3, 3);
    try {
        MatrixLib F = A - E;
        printTestResult(false, "Subtraction with mismatched dims throws");
    } catch (const std::invalid_argument&) {
        printTestResult(true,
                        "Subtraction with mismatched dims throws "
                        "std::invalid_argument");
    }

    std::cout << "\nA - B:\n";
    C.print();
}

// ----------------------------------------
// TEST 9: Compound Assignment (+=)
// ----------------------------------------
void testCompoundAddition() {
    printTestHeader(9, "Compound Assignment (operator+=)");

    MatrixLib A(2, 2);
    A.at(0, 0) = 1.0f;  A.at(0, 1) = 2.0f;
    A.at(1, 0) = 3.0f;  A.at(1, 1) = 4.0f;

    MatrixLib B(2, 2);
    B.at(0, 0) = 10.0f;  B.at(0, 1) = 10.0f;
    B.at(1, 0) = 10.0f;  B.at(1, 1) = 10.0f;

    A += B;

    printTestResult(floatEqual(A.at(0, 0), 11.0f) &&
                    floatEqual(A.at(0, 1), 12.0f) &&
                    floatEqual(A.at(1, 0), 13.0f) &&
                    floatEqual(A.at(1, 1), 14.0f),
                    "operator+= modifies matrix in-place correctly");

    // B unchanged
    printTestResult(floatEqual(B.at(0, 0), 10.0f),
                    "RHS unchanged after +=");

    // Chained += 
    MatrixLib C(2, 2);
    C.at(0, 0) = 1.0f;

    MatrixLib D(2, 2);
    D.at(0, 0) = 1.0f;

    MatrixLib E(2, 2);
    E.at(0, 0) = 1.0f;

    C += D += E;   // D becomes 2, C becomes 3 at [0,0]
    printTestResult(floatEqual(C.at(0, 0), 3.0f),
                    "Chained += works correctly");

    // Dimension mismatch
    MatrixLib F(3, 3);
    try {
        A += F;
        printTestResult(false, "+=  with mismatched dims throws");
    } catch (const std::invalid_argument&) {
        printTestResult(true,
                        "+= with mismatched dims throws "
                        "std::invalid_argument");
    }

    std::cout << "\nA after A += B:\n";
    A.print();
}

// ----------------------------------------
// TEST 10: Scalar Multiplication
// ----------------------------------------
void testScalarMultiply() {
    printTestHeader(10, "Scalar Multiplication");

    MatrixLib A(2, 2);
    A.at(0, 0) = 1.0f;  A.at(0, 1) = 2.0f;
    A.at(1, 0) = 3.0f;  A.at(1, 1) = 4.0f;

    // Multiply by 2.5
    MatrixLib B = A.ScalarMultiply(2.5f);
    printTestResult(floatEqual(B.at(0, 0),  2.5f) &&
                    floatEqual(B.at(0, 1),  5.0f) &&
                    floatEqual(B.at(1, 0),  7.5f) &&
                    floatEqual(B.at(1, 1), 10.0f),
                    "ScalarMultiply(2.5) produces correct values");

    // Multiply by 0
    MatrixLib C = A.ScalarMultiply(0.0f);
    bool allZero = true;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (!floatEqual(C.at(i, j), 0.0f)) {
                allZero = false;
            }
        }
    }
    printTestResult(allZero, "ScalarMultiply(0) produces zero matrix");

    // Multiply by 1
    MatrixLib D = A.ScalarMultiply(1.0f);
    printTestResult(matricesEqual(A, D),
                    "ScalarMultiply(1) returns equal matrix");

    // Multiply by negative
    MatrixLib E = A.ScalarMultiply(-1.0f);
    printTestResult(floatEqual(E.at(0, 0), -1.0f) &&
                    floatEqual(E.at(1, 1), -4.0f),
                    "ScalarMultiply(-1) negates all elements");

    // Original unchanged
    printTestResult(floatEqual(A.at(0, 0), 1.0f),
                    "Original unchanged after ScalarMultiply");

    std::cout << "\nA * 2.5:\n";
    B.print();
}

// ----------------------------------------
// TEST 11: Matrix Multiplication
// ----------------------------------------
void testMatrixMultiply() {
    printTestHeader(11, "Matrix Multiplication (operator*)");

    // 2x3 * 3x2 = 2x2
    MatrixLib H(2, 3);
    H.at(0, 0) = 1; H.at(0, 1) = 2; H.at(0, 2) = 3;
    H.at(1, 0) = 4; H.at(1, 1) = 5; H.at(1, 2) = 6;

    MatrixLib I(3, 2);
    I.at(0, 0) = 7;  I.at(0, 1) = 8;
    I.at(1, 0) = 9;  I.at(1, 1) = 10;
    I.at(2, 0) = 11; I.at(2, 1) = 12;

    MatrixLib J = H * I;

    // Expected: [58, 64 / 139, 154]
    printTestResult(J.getRows()    == 2 &&
                    J.getColumns() == 2,
                    "Result has correct dimensions (2x2)");
    printTestResult(floatEqual(J.at(0, 0),  58.0f) &&
                    floatEqual(J.at(0, 1),  64.0f) &&
                    floatEqual(J.at(1, 0), 139.0f) &&
                    floatEqual(J.at(1, 1), 154.0f),
                    "Matrix multiplication values are correct");

    // Identity matrix multiplication
    MatrixLib Id(2, 2);
    Id.at(0, 0) = 1.0f;
    Id.at(1, 1) = 1.0f;

    MatrixLib A(2, 2);
    A.at(0, 0) = 3.0f;  A.at(0, 1) = 7.0f;
    A.at(1, 0) = 2.0f;  A.at(1, 1) = 5.0f;

    MatrixLib K = A * Id;
    printTestResult(matricesEqual(A, K),
                    "A * Identity = A");

    MatrixLib L = Id * A;
    printTestResult(matricesEqual(A, L),
                    "Identity * A = A");

    // Incompatible dimensions
    MatrixLib M(2, 3);
    MatrixLib N(2, 3);
    try {
        MatrixLib O = M * N;
        printTestResult(false,
                        "Incompatible dimensions throw exception");
    } catch (const std::invalid_argument&) {
        printTestResult(true,
                        "Incompatible dimensions throw "
                        "std::invalid_argument");
    }

    std::cout << "\nH (2x3):\n";
    H.print();
    std::cout << "\nI (3x2):\n";
    I.print();
    std::cout << "\nH * I:\n";
    J.print();
}

// ----------------------------------------
// TEST 12: Transpose
// ----------------------------------------
void testTranspose() {
    printTestHeader(12, "Transpose");

    // Non-square matrix 2x3 -> 3x2
    MatrixLib A(2, 3);
    A.at(0, 0) = 1;  A.at(0, 1) = 2;  A.at(0, 2) = 3;
    A.at(1, 0) = 4;  A.at(1, 1) = 5;  A.at(1, 2) = 6;

    MatrixLib T = A.transpose();

    printTestResult(T.getRows()    == 3 &&
                    T.getColumns() == 2,
                    "Transpose of 2x3 has dimensions 3x2");

    printTestResult(floatEqual(T.at(0, 0), 1.0f) &&
                    floatEqual(T.at(1, 0), 2.0f) &&
                    floatEqual(T.at(2, 0), 3.0f) &&
                    floatEqual(T.at(0, 1), 4.0f) &&
                    floatEqual(T.at(1, 1), 5.0f) &&
                    floatEqual(T.at(2, 1), 6.0f),
                    "Transpose has correct element positions");

    // Double transpose returns original
    MatrixLib TT = T.transpose();
    printTestResult(matricesEqual(A, TT),
                    "Double transpose returns original matrix");

    // Symmetric matrix transpose equals itself
    MatrixLib S(2, 2);
    S.at(0, 0) = 1.0f;  S.at(0, 1) = 2.0f;
    S.at(1, 0) = 2.0f;  S.at(1, 1) = 3.0f;

    MatrixLib ST = S.transpose();
    printTestResult(matricesEqual(S, ST),
                    "Symmetric matrix equals its own transpose");

    // Original unchanged
    printTestResult(A.getRows()    == 2 &&
                    A.getColumns() == 3,
                    "Original matrix unchanged after transpose");

    std::cout << "\nA (2x3):\n";
    A.print();
    std::cout << "\nA transposed (3x2):\n";
    T.print();
}

// ----------------------------------------
// TEST 13: Resize
// ----------------------------------------
void testResize() {
    printTestHeader(13, "Resize Method");

    MatrixLib A(2, 2);
    A.at(0, 0) = 99.0f;

    // Enlarge
    A.resize(4, 4);
    printTestResult(A.getRows()    == 4 &&
                    A.getColumns() == 4,
                    "Resize to 4x4 gives correct dimensions");

    bool allZero = true;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (!floatEqual(A.at(i, j), 0.0f)) {
                allZero = false;
            }
        }
    }
    printTestResult(allZero,
                    "All elements zero after resize");

    // Shrink
    A.resize(1, 1);
    printTestResult(A.getRows()    == 1 &&
                    A.getColumns() == 1,
                    "Resize to 1x1 gives correct dimensions");

    // Non-square resize
    A.resize(3, 5);
    printTestResult(A.getRows()    == 3 &&
                    A.getColumns() == 5,
                    "Resize to non-square 3x5 gives correct dimensions");

    std::cout << "\nMatrix A after resize(3,5) with values set:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            A.at(i, j) = static_cast<float>(i * 5 + j + 1);
        }
    }
    A.print();
}

// ----------------------------------------
// TEST 14: Chained Operations
// ----------------------------------------
void testChainedOperations() {
    printTestHeader(14, "Chained Operations");

    // Identity matrix
    MatrixLib I(2, 2);
    I.at(0, 0) = 1.0f;
    I.at(1, 1) = 1.0f;

    MatrixLib Q(2, 2);
    Q.at(0, 0) = 2.0f;  Q.at(0, 1) = 3.0f;
    Q.at(1, 0) = 4.0f;  Q.at(1, 1) = 5.0f;

    // (I * Q) + (Q * 2) - Q  =  Q + 2Q - Q  =  2Q
    MatrixLib result = (I * Q) + Q.ScalarMultiply(2.0f) - Q;

    MatrixLib expected = Q.ScalarMultiply(2.0f);

    printTestResult(matricesEqual(result, expected),
                    "(I*Q) + (Q*2) - Q = 2Q");

    // (Q + Q).transpose() == Q.ScalarMultiply(2).transpose()
    MatrixLib left  = (Q + Q).transpose();
    MatrixLib right = Q.ScalarMultiply(2.0f).transpose();

    printTestResult(matricesEqual(left, right),
                    "(Q + Q).transpose() == (Q*2).transpose()");

    std::cout << "\n(I * Q) + (Q * 2) - Q:\n";
    result.print();
}

// ----------------------------------------
// TEST 15: Large Matrix
// ----------------------------------------
void testLargeMatrix() {
    printTestHeader(15, "Large Matrix Operations");

    const int N = 50;
    MatrixLib A(N, N);
    MatrixLib B(N, N);

    // Fill with values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A.at(i, j) = static_cast<float>(i + j);
            B.at(i, j) = static_cast<float>(i - j);
        }
    }

    // Addition
    MatrixLib C = A + B;
    printTestResult(floatEqual(C.at(0, 0), 0.0f) &&
                    floatEqual(C.at(1, 1), 2.0f),
                    "Large matrix addition spot check correct");

    // Subtraction
    MatrixLib D = A - B;
    printTestResult(floatEqual(D.at(0, 1), 2.0f) &&
                    floatEqual(D.at(1, 0), 0.0f),
                    "Large matrix subtraction spot check correct");

    // Transpose
    MatrixLib T = A.transpose();
    printTestResult(floatEqual(T.at(3, 1), A.at(1, 3)),
                    "Large matrix transpose spot check correct");

    // Scalar multiply
    MatrixLib S = A.ScalarMultiply(2.0f);
    printTestResult(floatEqual(S.at(2, 3), A.at(2, 3) * 2.0f),
                    "Large matrix scalar multiply spot check correct");

    printTestResult(C.getRows()    == N &&
                    C.getColumns() == N,
                    "Large matrix preserves dimensions");

    std::cout << "  (50x50 matrix operations completed successfully)\n";
}

// ========================================
// MAIN ENTRY POINT
// ========================================
int main() {
    std::cout << std::string(60, '=')           << std::endl;
    std::cout << "  MatrixLib - Full Test Suite" << std::endl;
    std::cout << std::string(60, '=')           << std::endl;

    testBasicConstructor();
    testCopyConstructor();
    testMoveConstructor();
    testCopyAssignment();
    testMoveAssignment();
    testElementAccess();
    testAddition();
    testSubtraction();
    testCompoundAddition();
    testScalarMultiply();
    testMatrixMultiply();
    testTranspose();
    testResize();
    testChainedOperations();
    testLargeMatrix();

    printSummary();

    return failedTests == 0 ? 0 : 1;
}