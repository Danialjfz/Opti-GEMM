#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include "MatrixLib.h"



volatile float sink = 0.0f;

template <typename Func>
long long time_ms(Func f)
{
    auto start = std::chrono::high_resolution_clock::now();

    f();  // run code

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
}

MatrixLib random_matrix(int r, int c)
{
    MatrixLib M(r, c);

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            // Cast to float before division to avoid integer division
            M.at(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    return M;
}

void benchmark_gemm(int size)
{
    MatrixLib A = random_matrix(size, size);
    MatrixLib B = random_matrix(size, size);

    // ----------------------------
    // WARMUP (IMPORTANT)
    // ----------------------------
    {
        MatrixLib C = A * B;
        sink += C.at(0, 0);
    }

    // ----------------------------
    // MEASUREMENTS
    // ----------------------------
    int iterations = 5;

    long long total = 0;
    long long best = LLONG_MAX;
    long long worst = 0;

    for (int i = 0; i < iterations; i++)
    {
        long long ms = time_ms([&]()
        {
            MatrixLib C = A * B;
            sink += C.at(0, 0);
        });

        total += ms;
        if (ms < best) best = ms;
        if (ms > worst) worst = ms;
    }

    long long avg = total / iterations;

    std::cout << size
              << ", avg=" << avg
              << ", best=" << best
              << ", worst=" << worst
              << "\n";
}



int main()
{
    // Seed the random number generator
    srand(static_cast<unsigned>(time(nullptr)));

    std::cout << "size,time_ms\n";

    int sizes[] = {128, 256, 512, 1024};

    for (int s : sizes)
    {
        benchmark_gemm(s);
    }

    return 0;
}