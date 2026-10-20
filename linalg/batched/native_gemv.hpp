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

// Internal header: GPU kernels for batched matrix-vector products, used by
// NativeBatchedLinAlg for a single right-hand side, and their tuning
// parameters. It is included by native.cpp and by the tuning benchmark
// config/tune_batched_gemv.cpp.
//
// The tuning parameters are the MFEM_GEMV_* macros below. Their defaults depend
// on the GPU vendor (CUDA or HIP build). To tune them for a specific GPU, run
//    python3 config/tune_batched_gemv.py --mfem-dir <MFEM build or install>
// and rebuild MFEM with the generated header given as MFEM_BATCHED_GEMV_TUNING
// (a CMake cache variable or a 'make config' variable). See INSTALL.

#ifndef MFEM_BATCHED_NATIVE_GEMV_HPP
#define MFEM_BATCHED_NATIVE_GEMV_HPP

#include "../../config/config.hpp"
#include "../../general/forall.hpp"

// The tuning header is installed next to config.hpp. While MFEM itself is
// built, MFEM_BATCHED_GEMV_TUNING_FILE points at the copy the build uses.
#if defined(MFEM_BATCHED_GEMV_TUNING) && !defined(MFEM_GEMV_IGNORE_TUNING)
#ifdef MFEM_BATCHED_GEMV_TUNING_FILE
#include MFEM_BATCHED_GEMV_TUNING_FILE
#else
#include "../../config/_batched_gemv_tuning.hpp"
#endif
#endif

#ifdef MFEM_GEMV_TUNED_FOR_HIP
#if (MFEM_GEMV_TUNED_FOR_HIP && !defined(MFEM_USE_HIP)) || \
    (!MFEM_GEMV_TUNED_FOR_HIP && defined(MFEM_USE_HIP))
#error "MFEM_BATCHED_GEMV_TUNING was generated for a different GPU backend"
#endif
#endif

// Tuning parameters. "Small" kernels handle m, n <= 32, "medium" kernels
// handle 32 < max(m, n) <= 128; N is y = A x and T is y = A^T x.
//
// MFEM_GEMV_SMALL_BLOCK_N, MFEM_GEMV_SMALL_BLOCK_T
//    Target threads per block of the small kernels (several matrices share a
//    block).
// MFEM_GEMV_SMALL_T_MAX
//    Largest max(m, n) for which T uses the small kernel; larger small
//    matrices use the medium T kernel. The small T kernel keeps a whole matrix
//    in shared memory, which can limit occupancy on GPUs with little shared
//    memory per compute unit.
// MFEM_GEMV_MED_N_ROUND, MFEM_GEMV_MED_N_BLOCK
//    Medium N: threads per matrix are the output size rounded up to a multiple
//    of ROUND, and a block has about BLOCK threads.
// MFEM_GEMV_MED_T_G1, MFEM_GEMV_MED_T_NT1, MFEM_GEMV_MED_T_SWITCH,
// MFEM_GEMV_MED_T_G2, MFEM_GEMV_MED_T_NT2
//    Medium T: one matrix per block of NT threads, with groups of G lanes
//    sharing each column. (G1, NT1) is used for max(m, n) <= SWITCH and
//    (G2, NT2) above.
#if defined(MFEM_USE_HIP)
// AMD Instinct GPUs (MI250X, MI300A/X, MI355X): 64-wide wavefronts, at most 32
// waves per compute unit, and 64 KiB of shared memory (LDS) per compute unit
// (160 KiB on MI355X). Not measured: chosen to fill whole wavefronts, to keep
// the shared memory per thread low, and to read 128-byte segments per column.
#ifndef MFEM_GEMV_SMALL_BLOCK_N
#define MFEM_GEMV_SMALL_BLOCK_N 128
#endif
#ifndef MFEM_GEMV_SMALL_BLOCK_T
#define MFEM_GEMV_SMALL_BLOCK_T 128
#endif
#ifndef MFEM_GEMV_SMALL_T_MAX
#define MFEM_GEMV_SMALL_T_MAX 16
#endif
#ifndef MFEM_GEMV_MED_N_ROUND
#define MFEM_GEMV_MED_N_ROUND 64
#endif
#ifndef MFEM_GEMV_MED_N_BLOCK
#define MFEM_GEMV_MED_N_BLOCK 256
#endif
#ifndef MFEM_GEMV_MED_T_G1
#define MFEM_GEMV_MED_T_G1 16
#endif
#ifndef MFEM_GEMV_MED_T_NT1
#define MFEM_GEMV_MED_T_NT1 256
#endif
#ifndef MFEM_GEMV_MED_T_SWITCH
#define MFEM_GEMV_MED_T_SWITCH 128
#endif
#ifndef MFEM_GEMV_MED_T_G2
#define MFEM_GEMV_MED_T_G2 16
#endif
#ifndef MFEM_GEMV_MED_T_NT2
#define MFEM_GEMV_MED_T_NT2 256
#endif
#else
// NVIDIA GPUs: measured on a GTX 1070 (sm_61). Compute capabilities 8.0
// (A100), 9.0 (H100, H200, GH200) and 10.0 (B200, GB200) have the same limits
// of 32 resident blocks and 64 resident warps per SM, and more shared memory
// per SM.
#ifndef MFEM_GEMV_SMALL_BLOCK_N
#define MFEM_GEMV_SMALL_BLOCK_N 64
#endif
#ifndef MFEM_GEMV_SMALL_BLOCK_T
#define MFEM_GEMV_SMALL_BLOCK_T 64
#endif
#ifndef MFEM_GEMV_SMALL_T_MAX
#define MFEM_GEMV_SMALL_T_MAX 32
#endif
#ifndef MFEM_GEMV_MED_N_ROUND
#define MFEM_GEMV_MED_N_ROUND 32
#endif
#ifndef MFEM_GEMV_MED_N_BLOCK
#define MFEM_GEMV_MED_N_BLOCK 256
#endif
#ifndef MFEM_GEMV_MED_T_G1
#define MFEM_GEMV_MED_T_G1 8
#endif
#ifndef MFEM_GEMV_MED_T_NT1
#define MFEM_GEMV_MED_T_NT1 128
#endif
#ifndef MFEM_GEMV_MED_T_SWITCH
#define MFEM_GEMV_MED_T_SWITCH 88
#endif
#ifndef MFEM_GEMV_MED_T_G2
#define MFEM_GEMV_MED_T_G2 16
#endif
#ifndef MFEM_GEMV_MED_T_NT2
#define MFEM_GEMV_MED_T_NT2 256
#endif
#endif

namespace mfem
{

namespace internal
{

namespace batched_gemv
{

// Batched matrix-vector products y_e = alpha op(A_e) x_e + beta y_e on a GPU,
// for column-major m x n matrices A_e stored contiguously (leading dimension
// m), one after another. y is not read when beta == 0.
//
// Except for the medium-size transpose, the kernels below map one thread to
// one output entry, so no cross-thread reductions are needed. Thread blocks
// are padded to a whole number of matrices, and inactive matrices are masked
// rather than returned from, so every thread reaches every MFEM_SYNC_THREAD.
// The kernels assume that each thread handles a single index of each
// MFEM_FOREACH_THREAD loop (values are kept in registers between loops), so
// they must run on a GPU, never through the host fallback of forall_2D.
// They do not depend on the warp or wavefront size.
//
// Design credits: packing several small matrices per block with one thread per
// output, staging x in shared memory, keeping a row of A in registers, and
// transposing small matrices through a padded shared-memory tile follow the
// approach of MAGMA's batched gemv (University of Tennessee, BSD-3) and of
// rocBLAS's small-matrix gemv (AMD, MIT). Groups of lanes per column with
// partial sums in shared memory, for the medium transpose, follow MAGMA's
// general transpose kernel and rocBLAS's. These ideas were taken from prose
// descriptions of the libraries; no code from either library is used.

/// Tuning parameters of DeviceGemv(), see the MFEM_GEMV_* macros above.
template <int SMALL_BLOCK_N_, int SMALL_BLOCK_T_, int SMALL_T_MAX_,
          int MED_N_ROUND_, int MED_N_BLOCK_, int MED_T_G1_, int MED_T_NT1_,
          int MED_T_SWITCH_, int MED_T_G2_, int MED_T_NT2_>
struct GemvParams
{
   static constexpr int SMALL_BLOCK_N = SMALL_BLOCK_N_;
   static constexpr int SMALL_BLOCK_T = SMALL_BLOCK_T_;
   static constexpr int SMALL_T_MAX = SMALL_T_MAX_;
   static constexpr int MED_N_ROUND = MED_N_ROUND_;
   static constexpr int MED_N_BLOCK = MED_N_BLOCK_;
   static constexpr int MED_T_G1 = MED_T_G1_;
   static constexpr int MED_T_NT1 = MED_T_NT1_;
   static constexpr int MED_T_SWITCH = MED_T_SWITCH_;
   static constexpr int MED_T_G2 = MED_T_G2_;
   static constexpr int MED_T_NT2 = MED_T_NT2_;
};

using DefaultGemvParams =
   GemvParams<MFEM_GEMV_SMALL_BLOCK_N, MFEM_GEMV_SMALL_BLOCK_T,
   MFEM_GEMV_SMALL_T_MAX, MFEM_GEMV_MED_N_ROUND, MFEM_GEMV_MED_N_BLOCK,
   MFEM_GEMV_MED_T_G1, MFEM_GEMV_MED_T_NT1, MFEM_GEMV_MED_T_SWITCH,
   MFEM_GEMV_MED_T_G2, MFEM_GEMV_MED_T_NT2>;

/// Static shared memory per block is kept below this (the CUDA limit without
/// opt-in; AMD GPUs have at least 64 KiB).
constexpr int GEMV_SMEM_BYTES = 48 * 1024;

/// Number of matrices of size at most @a n handled by one thread block, so
/// that a block has roughly @a threads threads. The matrices are the z
/// dimension of the block, which CUDA limits to 64.
constexpr int GemvBatchPerBlock(int n, int threads)
{
   return n >= threads ? 1 : (threads / n > 64 ? 64 : threads / n);
}

constexpr int GemvRoundUp(int n, int r) { return (n + r - 1) / r * r; }

/// Matrices with m, n <= 32. A whole matrix is handled by one row of threads,
/// and several matrices share a block of about BLOCK threads (fewer if the
/// shared memory would exceed GEMV_SMEM_BYTES). x_e is staged in shared memory.
/// Thread t first loads row t of A_e into registers. Consecutive threads read
/// consecutive entries of each column, so the loads are coalesced, and all the
/// loads of a thread are issued before the first barrier, so many are in
/// flight at once.
/// NoTrans: thread i then computes its output from the row in registers.
/// Trans: the rows are written to a shared tile with an odd row pitch, and
/// thread i reads column i of A_e back from the tile; on NVIDIA GPUs neither
/// the tile writes nor the reads have bank conflicts.
/// T_M and T_N give the sizes at compile time; when they are 0, the runtime
/// sizes m and n are used with T_MAX as the bound. The loops always run to the
/// compile-time bound (with a guard in the runtime case) so that they unroll
/// fully and the row stays in registers.
/// (On a GTX 1070, issuing all the loads before the barrier and using blocks of
/// about 64 rather than 128 threads each raised the Trans bandwidth for n = 18
/// to 31 from 170-195 to 205-210 GB/s.)
template <bool TR, int T_M, int T_N, int T_MAX, int BLOCK>
void SmallGemv(const int m_, const int n_, const int n_mat, const real_t *A,
               const real_t *x, real_t *y, const real_t alpha,
               const real_t beta)
{
   static_assert(T_MAX <= 32, "too large for the small-matrix kernel");
   constexpr int MAX_M = T_M ? T_M : T_MAX;
   constexpr int MAX_N = T_N ? T_N : T_MAX;
   constexpr int MAX_IN = TR ? MAX_M : MAX_N;
   constexpr int MAX_OUT = TR ? MAX_N : MAX_M;
   constexpr int MAX_T = MAX_M > MAX_N ? MAX_M : MAX_N; // threads per matrix
   constexpr int PITCH = MAX_N | 1;
   constexpr int MAT_BYTES = int(sizeof(real_t)) *
                             (MAX_IN + (TR ? MAX_M * PITCH : 1));
   constexpr int C0 = GemvBatchPerBlock(MAX_T, BLOCK);
   constexpr int C1 = GEMV_SMEM_BYTES / MAT_BYTES;
   constexpr int C = C0 < C1 ? C0 : (C1 > 0 ? C1 : 1);
   const int m = T_M ? T_M : m_;
   const int n = T_N ? T_N : n_;
   const int n_in = TR ? m : n, n_out = TR ? n : m;
   const int n_pad = (n_mat + C - 1) / C * C;
   mfem::forall_2D_batch(n_pad, MAX_T, 1, C, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_SHARED real_t s_x[C][MAX_IN];
      MFEM_SHARED real_t s_A[C][TR ? MAX_M : 1][TR ? PITCH : 1];
      const int tz = MFEM_THREAD_ID(z);
      const bool active = e < n_mat;
      const size_t e_ = active ? e : 0;
      const real_t *A_e = A + e_ * m * n;
      const real_t *x_e = x + e_ * n_in;
      real_t *y_e = y + e_ * n_out;
      real_t a[MAX_N]; // row t of A_e
      MFEM_FOREACH_THREAD(t, x, MAX_T)
      {
         if (active && t < n_in) { s_x[tz][t] = x_e[t]; }
         if (active && t < m)
         {
            MFEM_UNROLL(MAX_N)
            for (int j = 0; j < MAX_N; j++)
            {
               if (j < n) { a[j] = A_e[t + j * m]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (TR)
      {
         MFEM_FOREACH_THREAD(t, x, MAX_T)
         {
            if (active && t < m)
            {
               MFEM_UNROLL(MAX_N)
               for (int j = 0; j < MAX_N; j++)
               {
                  if (j < n) { s_A[tz][t][j] = a[j]; }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_FOREACH_THREAD(i, x, MAX_OUT)
      {
         if (active && i < n_out)
         {
            real_t dot = 0.0;
            MFEM_UNROLL(MAX_IN)
            for (int j = 0; j < MAX_IN; j++)
            {
               if (j < n_in)
               {
                  if (TR) { dot += s_A[tz][j][i] * s_x[tz][j]; }
                  else { dot += a[j] * s_x[tz][j]; }
               }
            }
            y_e[i] = (beta == 0.0) ? alpha * dot : alpha * dot + beta * y_e[i];
         }
      }
   });
}

/// Square matrices of compile-time size, N <= 32.
template <bool TR, int BLOCK, int N = 1>
void SmallSquareGemv(const int n, const int n_mat, const real_t *A,
                     const real_t *x, real_t *y, const real_t alpha,
                     const real_t beta)
{
   if constexpr (N > 32) { MFEM_ABORT("unreachable"); }
   else if (n == N)
   {
      SmallGemv<TR, N, N, 32, BLOCK>(n, n, n_mat, A, x, y, alpha, beta);
   }
   else { SmallSquareGemv<TR, BLOCK, N + 1>(n, n_mat, A, x, y, alpha, beta); }
}

/// NoTrans with max(m, n) <= T_MAX: one thread per row with direct coalesced
/// loads, several matrices per block, x_e staged in shared memory. Threads per
/// matrix are m rounded up to a multiple of ROUND.
template <int T_MAX, int ROUND, int BLOCK>
void MediumGemvN(const int m, const int n, const int n_mat, const real_t *A,
                 const real_t *x, real_t *y, const real_t alpha,
                 const real_t beta)
{
   constexpr int C = GemvBatchPerBlock(GemvRoundUp(T_MAX, ROUND), BLOCK);
   const int X = GemvRoundUp(m, ROUND);
   const int n_pad = (n_mat + C - 1) / C * C;
   mfem::forall_2D_batch(n_pad, X, 1, C, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_SHARED real_t s_x[C][T_MAX];
      const int tz = MFEM_THREAD_ID(z);
      const bool active = e < n_mat;
      const size_t e_ = active ? e : 0;
      const real_t *A_e = A + e_ * m * n;
      const real_t *x_e = x + e_ * n;
      real_t *y_e = y + e_ * m;
      MFEM_FOREACH_THREAD(t, x, n)
      {
         if (active) { s_x[tz][t] = x_e[t]; }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i, x, m)
      {
         if (active)
         {
            real_t dot = 0.0;
            for (int j = 0; j < n; j++) { dot += A_e[i + j * m] * s_x[tz][j]; }
            y_e[i] = (beta == 0.0) ? alpha * dot : alpha * dot + beta * y_e[i];
         }
      }
   });
}

/// Trans with max(m, n) <= T_MAX: one matrix per block of NT threads. Groups
/// of G lanes share a column, each lane summing a strided subset of its rows,
/// so that the loads of a group are contiguous; the partial sums are added in
/// shared memory.
/// (On a GTX 1070, G = 8 with NT = 128 measured fastest for n up to about 88;
/// above that, G = 16 with NT = 256 is as fast or faster, and avoids a drop
/// from about 210 to 150-175 GB/s that G = 8 with NT = 128 shows for n = 121 to
/// 127.)
template <int T_MAX, int G, int NT>
void MediumGemvT(const int m, const int n, const int n_mat, const real_t *A,
                 const real_t *x, real_t *y, const real_t alpha,
                 const real_t beta)
{
   static_assert(NT % G == 0 && NT <= 1024, "invalid G or NT");
   static_assert(int(sizeof(real_t)) * T_MAX * (G + 2) <= GEMV_SMEM_BYTES,
                 "too much shared memory");
   // G lanes share each column: lane g sums rows g, g + G, ... of the column
   // in a register, so G consecutive lanes read G consecutive entries. The G
   // partial sums are combined through shared memory. The Y rows of lanes
   // take columns ty, ty + Y, ...
   constexpr int Y = NT / G;
   mfem::forall_2D(n_mat, G, Y, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_SHARED real_t s_x[T_MAX];
      MFEM_SHARED real_t s_part[T_MAX][G + 1];
      const real_t *A_e = A + size_t(e) * m * n;
      const real_t *x_e = x + size_t(e) * m;
      real_t *y_e = y + size_t(e) * n;
      MFEM_FOREACH_THREAD(ty, y, Y)
      {
         MFEM_FOREACH_THREAD(tx, x, G)
         {
            for (int r = tx + G * ty; r < m; r += G * Y) { s_x[r] = x_e[r]; }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(ty, y, Y)
      {
         MFEM_FOREACH_THREAD(tx, x, G)
         {
            for (int j = ty; j < n; j += Y)
            {
               const real_t *A_j = A_e + size_t(j) * m;
               real_t part = 0.0;
               for (int r = tx; r < m; r += G) { part += A_j[r] * s_x[r]; }
               s_part[j][tx] = part;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(ty, y, Y)
      {
         MFEM_FOREACH_THREAD(tx, x, G)
         {
            for (int j = tx + G * ty; j < n; j += G * Y)
            {
               real_t dot = 0.0;
               for (int g = 0; g < G; g++) { dot += s_part[j][g]; }
               y_e[j] = (beta == 0.0) ? alpha * dot : alpha * dot + beta * y_e[j];
            }
         }
      }
   });
}

/// MediumGemvN with the smallest instantiated T_MAX that fits, max(m, n) <= 128.
template <int ROUND, int BLOCK>
void MediumGemvNSized(const int m, const int n, const int n_mat,
                      const real_t *A, const real_t *x, real_t *y,
                      const real_t alpha, const real_t beta)
{
   if (m <= 64 && n <= 64)
   {
      MediumGemvN<64, ROUND, BLOCK>(m, n, n_mat, A, x, y, alpha, beta);
   }
   else { MediumGemvN<128, ROUND, BLOCK>(m, n, n_mat, A, x, y, alpha, beta); }
}

/// MediumGemvT with the smallest instantiated T_MAX that fits, max(m, n) <= 128.
template <int G, int NT>
void MediumGemvTSized(const int m, const int n, const int n_mat,
                      const real_t *A, const real_t *x, real_t *y,
                      const real_t alpha, const real_t beta)
{
   const int big = m > n ? m : n;
   if (big <= 32) { MediumGemvT<32, G, NT>(m, n, n_mat, A, x, y, alpha, beta); }
   else if (big <= 64) { MediumGemvT<64, G, NT>(m, n, n_mat, A, x, y, alpha, beta); }
   else if (big <= 96) { MediumGemvT<96, G, NT>(m, n, n_mat, A, x, y, alpha, beta); }
   else { MediumGemvT<128, G, NT>(m, n, n_mat, A, x, y, alpha, beta); }
}

/// Run the kernel that P selects for op(A_e) with A_e of size m x n. Returns
/// false if no kernel applies (max(m, n) > 128).
template <typename P = DefaultGemvParams>
bool DeviceGemv(bool tr, int m, int n, int n_mat, const real_t *A,
                const real_t *x, real_t *y, real_t alpha, real_t beta)
{
   static_assert(P::SMALL_BLOCK_N >= 1 && P::SMALL_BLOCK_N <= 1024 &&
                 P::SMALL_BLOCK_T >= 1 && P::SMALL_BLOCK_T <= 1024 &&
                 P::SMALL_T_MAX >= 0 && P::SMALL_T_MAX <= 32 &&
                 P::MED_N_ROUND >= 1 && P::MED_N_BLOCK >= 1 &&
                 GemvRoundUp(128, P::MED_N_ROUND) *
                 GemvBatchPerBlock(GemvRoundUp(128, P::MED_N_ROUND),
                                   P::MED_N_BLOCK) <= 1024,
                 "invalid MFEM_GEMV_* tuning parameters");
   const int big = (m > n) ? m : n;
   if (big > 128) { return false; }
   if (tr && (big > 32 || big > P::SMALL_T_MAX))
   {
      if (big <= P::MED_T_SWITCH)
      {
         MediumGemvTSized<P::MED_T_G1, P::MED_T_NT1>(m, n, n_mat, A, x, y,
                                                    alpha, beta);
      }
      else
      {
         MediumGemvTSized<P::MED_T_G2, P::MED_T_NT2>(m, n, n_mat, A, x, y,
                                                    alpha, beta);
      }
   }
   else if (big > 32)
   {
      MediumGemvNSized<P::MED_N_ROUND, P::MED_N_BLOCK>(m, n, n_mat, A, x, y,
                                                       alpha, beta);
   }
   else if (m == n)
   {
      if (tr)
      {
         SmallSquareGemv<true, P::SMALL_BLOCK_T>(n, n_mat, A, x, y, alpha, beta);
      }
      else
      {
         SmallSquareGemv<false, P::SMALL_BLOCK_N>(n, n_mat, A, x, y, alpha, beta);
      }
   }
   else if (tr)
   {
      SmallGemv<true, 0, 0, 32, P::SMALL_BLOCK_T>(m, n, n_mat, A, x, y, alpha,
                                                  beta);
   }
   else
   {
      SmallGemv<false, 0, 0, 32, P::SMALL_BLOCK_N>(m, n, n_mat, A, x, y, alpha,
                                                   beta);
   }
   return true;
}

} // namespace batched_gemv

} // namespace internal

} // namespace mfem

#endif
