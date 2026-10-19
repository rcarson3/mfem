// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Benchmark of the candidate parameters of the native batched gemv kernels in
// linalg/batched/native_gemv.hpp. Built and run by tune_batched_gemv.py, which
// compiles this file several times with different TUNE_INST_* macros (one
// object per group of kernel instantiations, so that they compile in
// parallel) and once with TUNE_MAIN.
//
// Macros set by the script:
//    TUNE_SMALL_LIST   X(B) for each small-kernel block size B
//    TUNE_MEDT_LIST    X(1000 G + NT) for each medium-Trans configuration
//    TUNE_MEDN_LIST    X(1000 R + B) for each medium-NoTrans configuration
//    TUNE_INST_SMALL   B: instantiate the small kernels for block size B
//    TUNE_INST_MEDT    X(1000 G + NT)...: instantiate these medium-Trans kernels
//    TUNE_INST_MEDN    X(1000 R + B)...: instantiate these medium-NoTrans kernels
// (Configurations are encoded as one number because nvcc splits -D values at
// commas.)
//    TUNE_MAIN         compile main()
//
// Usage: tune_batched_gemv <budget MB> <trial ms> <vendor 0/1> <stride>
// Output lines (whitespace separated):
//    INFO <key> <value...>
//    DEFAULT <name>=<value> ...
//    R <family> <config> <op> <n> <n_mat> <us/call> <GB/s> <max error>
// where GB/s counts the bytes of A, x and y.

#define MFEM_GEMV_IGNORE_TUNING
#include "mfem.hpp"
#include "native_gemv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace mfem;
namespace bg = mfem::internal::batched_gemv;

template <bool TR, int B>
void RunSmall(int n, int n_mat, const real_t *A, const real_t *x, real_t *y);
template <int C>
void RunMedT(int n, int n_mat, const real_t *A, const real_t *x, real_t *y);
template <int C>
void RunMedN(int n, int n_mat, const real_t *A, const real_t *x, real_t *y);

#ifdef TUNE_INST_SMALL
template <bool TR, int B>
void RunSmall(int n, int n_mat, const real_t *A, const real_t *x, real_t *y)
{
   bg::SmallSquareGemv<TR, B>(n, n_mat, A, x, y, 1.0, 0.0);
}
template void RunSmall<false, TUNE_INST_SMALL>(int, int, const real_t *,
                                               const real_t *, real_t *);
template void RunSmall<true, TUNE_INST_SMALL>(int, int, const real_t *,
                                              const real_t *, real_t *);
#endif

#ifdef TUNE_INST_MEDT
template <int C>
void RunMedT(int n, int n_mat, const real_t *A, const real_t *x, real_t *y)
{
   bg::MediumGemvTSized<C / 1000, C % 1000>(n, n, n_mat, A, x, y, 1.0, 0.0);
}
#define X(C) template void RunMedT<C>(int, int, const real_t *, \
                                      const real_t *, real_t *);
TUNE_INST_MEDT
#undef X
#endif

#ifdef TUNE_INST_MEDN
template <int C>
void RunMedN(int n, int n_mat, const real_t *A, const real_t *x, real_t *y)
{
   bg::MediumGemvNSized<C / 1000, C % 1000>(n, n, n_mat, A, x, y, 1.0, 0.0);
}
#define X(C) template void RunMedN<C>(int, int, const real_t *, \
                                      const real_t *, real_t *);
TUNE_INST_MEDN
#undef X
#endif

#ifdef TUNE_MAIN

using Clock = std::chrono::steady_clock;

struct Bench
{
   double budget; // bytes of A + x + y per test
   double trial_s; // minimum duration of one timing trial
   Vector A, x, y, y_ref, err;

   int NumMat(int n) const { return int(budget / (sizeof(real_t) * (double(n) * n + 2 * n))); }

   // Max |y - y_ref| over the first n_out * n_mat entries.
   double Error(int n_out, int n_mat)
   {
      const int N = n_out * n_mat, chunk = 4096;
      const int nc = (N + chunk - 1) / chunk;
      const real_t *d_y = y.Read(), *d_r = y_ref.Read();
      real_t *d_e = err.Write();
      mfem::forall(nc, [=] MFEM_HOST_DEVICE (int c)
      {
         real_t e = 0.0;
         const int end = (c + 1) * chunk < N ? (c + 1) * chunk : N;
         for (int i = c * chunk; i < end; i++)
         {
            const real_t d = fabs(d_y[i] - d_r[i]);
            e = d > e ? d : e;
         }
         d_e[c] = e;
      });
      const real_t *h_e = err.HostRead();
      double e = 0.0;
      for (int c = 0; c < nc; c++) { e = std::max(e, double(h_e[c])); }
      return e;
   }

   // Reference result in y_ref, one thread per output entry.
   void Reference(bool tr, int n, int n_mat)
   {
      const real_t *d_A = A.Read(), *d_x = x.Read();
      real_t *d_r = y_ref.Write();
      mfem::forall(n * n_mat, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int i = idx % n, e = idx / n;
         const real_t *A_e = d_A + size_t(e) * n * n;
         real_t dot = 0.0;
         for (int j = 0; j < n; j++)
         {
            dot += (tr ? A_e[j + i * n] : A_e[i + j * n]) * d_x[size_t(e) * n + j];
         }
         d_r[idx] = dot;
      });
   }

   // Time f() and print a result line.
   template <typename F>
   void Time(const char *family, const std::string &cfg, bool tr, int n,
             int n_mat, F &&f)
   {
      f(); // warm up, and check
      MFEM_DEVICE_SYNC;
      const double error = Error(n, n_mat);
      auto t0 = Clock::now();
      f();
      MFEM_DEVICE_SYNC;
      const double t1 = std::chrono::duration<double>(Clock::now() - t0).count();
      const int reps = std::max(2, std::min(200, int(trial_s / std::max(t1, 1e-7))));
      double best = t1;
      for (int trial = 0; trial < 3; trial++)
      {
         MFEM_DEVICE_SYNC;
         t0 = Clock::now();
         for (int r = 0; r < reps; r++) { f(); }
         MFEM_DEVICE_SYNC;
         best = std::min(best, std::chrono::duration<double>(Clock::now() - t0).count() / reps);
      }
      const double bytes = sizeof(real_t) * (double(n) * n + 2.0 * n) * n_mat;
      printf("R %s %s %s %d %d %.2f %.2f %.3e\n", family, cfg.c_str(),
             tr ? "T" : "N", n, n_mat, best * 1e6, bytes / best * 1e-9, error);
      fflush(stdout);
   }
};

static void PrintGpuInfo()
{
#if defined(MFEM_USE_CUDA)
   int dev = 0;
   MFEM_GPU_CHECK(cudaGetDevice(&dev));
   cudaDeviceProp p;
   MFEM_GPU_CHECK(cudaGetDeviceProperties(&p, dev));
   printf("INFO vendor NVIDIA\n");
   printf("INFO name %s\n", p.name);
   printf("INFO arch sm_%d%d\n", p.major, p.minor);
   printf("INFO max_blocks_per_cu %d\n", p.maxBlocksPerMultiProcessor);
   printf("INFO shared_per_cu %zu\n", size_t(p.sharedMemPerMultiprocessor));
#elif defined(MFEM_USE_HIP)
   int dev = 0;
   MFEM_GPU_CHECK(hipGetDevice(&dev));
   hipDeviceProp_t p;
   MFEM_GPU_CHECK(hipGetDeviceProperties(&p, dev));
   printf("INFO vendor AMD\n");
   printf("INFO name %s\n", p.name);
   printf("INFO arch %s\n", p.gcnArchName);
   printf("INFO shared_per_cu %zu\n", size_t(p.maxSharedMemoryPerMultiProcessor));
#else
#error "MFEM must be built with CUDA or HIP"
#endif
   printf("INFO compute_units %d\n", p.multiProcessorCount);
   printf("INFO warp_size %d\n", p.warpSize);
   printf("INFO max_threads_per_cu %d\n", p.maxThreadsPerMultiProcessor);
   printf("INFO l2_bytes %d\n", p.l2CacheSize);
   printf("INFO real_t_bytes %d\n", int(sizeof(real_t)));
}

static bool Strided(int n, int stride)
{
   // With stride > 1, keep every stride-th size plus the element sizes of
   // low-order (p <= 2) scalar and vector H1/L2 spaces.
   static const int fe[] = {1, 2, 3, 4, 6, 8, 9, 10, 12, 18, 24, 27, 30, 54, 81};
   for (int s : fe) { if (n == s) { return true; } }
   return stride <= 1 || n % stride == 0;
}

int main(int argc, char *argv[])
{
#if defined(MFEM_USE_HIP)
   Device device("hip");
#else
   Device device("cuda");
#endif
   Bench b;
   b.budget = 1e6 * (argc > 1 ? atof(argv[1]) : 400.0);
   b.trial_s = 1e-3 * (argc > 2 ? atof(argv[2]) : 20.0);
   const bool vendor = argc > 3 ? atoi(argv[3]) != 0 : true;
   const int stride = argc > 4 ? atoi(argv[4]) : 1;

   PrintGpuInfo();
   printf("INFO budget_mb %.0f\n", b.budget * 1e-6);
   using P = bg::DefaultGemvParams;
   printf("DEFAULT SMALL_BLOCK_N=%d SMALL_BLOCK_T=%d SMALL_T_MAX=%d "
          "MED_N_ROUND=%d MED_N_BLOCK=%d MED_T_G1=%d MED_T_NT1=%d "
          "MED_T_SWITCH=%d MED_T_G2=%d MED_T_NT2=%d\n", P::SMALL_BLOCK_N,
          P::SMALL_BLOCK_T, P::SMALL_T_MAX, P::MED_N_ROUND, P::MED_N_BLOCK,
          P::MED_T_G1, P::MED_T_NT1, P::MED_T_SWITCH, P::MED_T_G2, P::MED_T_NT2);
   fflush(stdout);

   // Buffers sized for the largest A and x over n = 1..128.
   size_t a_max = 0, x_max = 0;
   for (int n = 1; n <= 128; n++)
   {
      a_max = std::max(a_max, size_t(n) * n * b.NumMat(n));
      x_max = std::max(x_max, size_t(n) * b.NumMat(n));
   }
   b.A.SetSize(int(a_max)); b.x.SetSize(int(x_max));
   b.y.SetSize(int(x_max)); b.y_ref.SetSize(int(x_max));
   b.err.SetSize(int(x_max / 4096 + 1));
   for (Vector *v : {&b.A, &b.x, &b.y, &b.y_ref, &b.err}) { v->UseDevice(true); }
   {
      real_t *d_A = b.A.Write(), *d_x = b.x.Write();
      mfem::forall(int(a_max), [=] MFEM_HOST_DEVICE (int i)
      {
         d_A[i] = 1.0 / (1 + i % 97);
      });
      mfem::forall(int(x_max), [=] MFEM_HOST_DEVICE (int i)
      {
         d_x[i] = 0.5 - ((i * 7919) % 1000) * 1e-3;
      });
      b.y = 0.0;
   }

   const bool magma = BatchedLinAlg::IsAvailable(BatchedLinAlg::MAGMA);
   for (int tr = 0; tr <= 1; tr++)
   {
      for (int n = 1; n <= 128; n++)
      {
         const bool small = n <= 32;
         const bool medt = tr && n >= 9 && (n <= 32 || Strided(n, stride));
         const bool medn = !tr && n > 32 && Strided(n, stride);
         const bool vend = vendor && Strided(n, 8);
         if (!small && !medt && !medn && !vend) { continue; }
         const int n_mat = b.NumMat(n);
         b.Reference(tr, n, n_mat);
         const real_t *A = b.A.Read(), *x = b.x.Read();
         real_t *y = b.y.Write();
         if (small)
         {
#define X(B) b.Time("small", std::to_string(B), tr, n, n_mat, [&]() \
         { if (tr) { RunSmall<true, B>(n, n_mat, A, x, y); }              \
           else { RunSmall<false, B>(n, n_mat, A, x, y); } });
            TUNE_SMALL_LIST
#undef X
         }
         if (medt)
         {
#define X(C) b.Time("medt", std::to_string(C / 1000) + "/" +             \
         std::to_string(C % 1000), tr, n, n_mat,                            \
         [&]() { RunMedT<C>(n, n_mat, A, x, y); });
            TUNE_MEDT_LIST
#undef X
         }
         if (medn)
         {
#define X(C) b.Time("medn", std::to_string(C / 1000) + "/" +             \
         std::to_string(C % 1000), tr, n, n_mat,                            \
         [&]() { RunMedN<C>(n, n_mat, A, x, y); });
            TUNE_MEDN_LIST
#undef X
         }
         if (vend)
         {
            // Vendor libraries through BatchedLinAlg, on aliases of the
            // same data.
            DenseTensor At;
            At.NewMemoryAndSize(Memory<real_t>(b.A.GetMemory(), 0, n * n * n_mat),
                                n, n, n_mat, false);
            Vector xs(b.x, 0, n * n_mat), ys(b.y, 0, n * n_mat);
            const auto op = tr ? BatchedLinAlg::Op::T : BatchedLinAlg::Op::N;
            b.Time("vendor", "GPU_BLAS", tr, n, n_mat, [&]()
            {
               BatchedLinAlg::Get(BatchedLinAlg::GPU_BLAS).AddMult(At, xs, ys, 1.0,
                                                                   0.0, op);
            });
            if (magma)
            {
               b.Time("vendor", "MAGMA", tr, n, n_mat, [&]()
               {
                  BatchedLinAlg::Get(BatchedLinAlg::MAGMA).AddMult(At, xs, ys, 1.0,
                                                                   0.0, op);
               });
            }
         }
      }
   }
   printf("INFO done 1\n");
   return 0;
}

#endif // TUNE_MAIN
