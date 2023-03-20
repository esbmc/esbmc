#if !defined(CURAND_KERNEL_H_)
#define CURAND_KERNEL_H_

//#include "curand.h"
#include "curand_precalc.h"
#include <math.h>
#include <curand.h>

#define MAX_XOR_N (5)
#define SKIPAHEAD_BLOCKSIZE (4)
#define SKIPAHEAD_MASK ((1 << SKIPAHEAD_BLOCKSIZE) - 1)
#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_DOUBLE (2.3283064365386963e-10)
#define CURAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define CURAND_2POW53_INV_2PI_DOUBLE                                           \
  (1.1102230246251565e-16 * 6.2831853071795860)
#define CURAND_SQRT2 (-1.4142135f)
#define CURAND_SQRT2_DOUBLE (-1.4142135623730951)

//Insert of curand.h
//typedef unsigned long long curandDirectionVectors64_t[64];
//typedef unsigned int curandDirectionVectors32_t[32];

//defines of curand_precalc.h
#define PRECALC_NUM_MATRICES (8)
#define PRECALC_BLOCK_SIZE (2)
#define PRECALC_BLOCK_MASK ((1 << PRECALC_BLOCK_SIZE) - 1)
#define XORWOW_SEQUENCE_SPACING (67)

static unsigned int precalc_xorwow_matrix[8][2] = {{850664906UL, 4258393217UL}};

#if !defined(QUALIFIERS)
#define QUALIFIERS static inline __device__
#endif

/* Test RNG */
/* This generator uses the formula:
   x_n = x_(n-1) + 1 mod 2^32
   x_0 = (unsigned int)seed * 3
   Subsequences are spaced 31337 steps apart.
*/
struct curandStateTest
{
  unsigned int v;
};

typedef struct curandStateTest curandStateTest_t;

/* XORSHIFT FAMILY RNGs */
/* These generators are a family proposed by Marsaglia.  They keep state
   in 32 bit chunks, then use repeated shift and xor operations to scramble
   the bits.  The following generators are a combination of a simple Weyl
   generator with an N variable XORSHIFT generator.
*/

/* XORSHIFT RNG */
/* This generator uses the xorwow formula of
www.jstatsoft.org/v08/i14/paper page 5
Has period 2^192 - 2^32.
*/
/**
 * CURAND XORWOW state
 */
struct curandStateXORWOW;

/**
 * CURAND XORWOW state
 */
typedef struct curandStateXORWOW curandStateXORWOW_t;

/* Implementation details not in reference documentation */
struct curandStateXORWOW
{
  unsigned int d, v[5];
  int boxmuller_flag;
  float boxmuller_extra;
  double boxmuller_extra_double;
};

/* SOBOL QRNG */
/**
 * CURAND Sobol32 state
 */
struct curandStateSobol32;

/* Implementation details not in reference documentation */
struct curandStateSobol32
{
  unsigned int i, x;
  unsigned int direction_vectors[32];
};

/**
 * CURAND Sobol32 state
 */
typedef struct curandStateSobol32 curandStateSobol32_t;

/**
 * CURAND Scrambled Sobol32 state
 */
struct curandStateScrambledSobol32;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol32
{
  unsigned int i, x, c;
  unsigned int direction_vectors[32];
};

/**
 * CURAND Scrambled Sobol32 state
 */
typedef struct curandStateScrambledSobol32 curandStateScrambledSobol32_t;

/**
 * CURAND Sobol64 state
 */
struct curandStateSobol64;

/* Implementation details not in reference documentation */
struct curandStateSobol64
{
  unsigned long long i, x;
  unsigned long long direction_vectors[64];
};

/**
 * CURAND Sobol64 state
 */
typedef struct curandStateSobol64 curandStateSobol64_t;

/**
 * CURAND Scrambled Sobol64 state
 */
struct curandStateScrambledSobol64;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol64
{
  unsigned long long i, x, c;
  unsigned long long direction_vectors[64];
};

/**
 * CURAND Scrambled Sobol64 state
 */
typedef struct curandStateScrambledSobol64 curandStateScrambledSobol64_t;

/**
 * Default RNG
 */
typedef struct curandStateXORWOW curandState_t;
typedef struct curandStateXORWOW curandState;

/****************************************************************************/
/* Utility functions needed by RNGs */
/****************************************************************************/

/* multiply vector by matrix, store in result
   matrix is n x n, measured in 32 bit units
   matrix is stored in row major order
   vector and result cannot be same pointer
*/
/*QUALIFIERS*/ void __curand_matvec(
  unsigned int *vector,
  unsigned int *matrix,
  unsigned int *result,
  int n)
{
  for(int i = 0; i < n; i++)
  {
    result[i] = 0;
  }
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < 32; j++)
    {
      if(vector[i] & (1 << j))
      {
        for(int k = 0; k < n; k++)
        {
          result[k] ^= matrix[n * (i * 32 + j) + k];
        }
      }
    }
  }
}

/* generate identity matrix */
/*QUALIFIERS*/ void __curand_matidentity(unsigned int *matrix, int n)
{
  int r;
  for(int i = 0; i < n * 32; i++)
  {
    for(int j = 0; j < n; j++)
    {
      r = i & 31;
      if(i / 32 == j)
      {
        matrix[i * n + j] = (1 << r);
      }
      else
      {
        matrix[i * n + j] = 0;
      }
    }
  }
}

/* multiply matrixA by matrixB, store back in matrixA
   matrixA and matrixB must not be same matrix */
/*QUALIFIERS*/ void
__curand_matmat(unsigned int *matrixA, unsigned int *matrixB, int n)
{
  unsigned int result[MAX_XOR_N];
  for(int i = 0; i < n * 32; i++)
  {
    __curand_matvec(matrixA + i * n, matrixB, result, n);
    for(int j = 0; j < n; j++)
    {
      matrixA[i * n + j] = result[j];
    }
  }
}

/* copy vectorA to vector */
/*QUALIFIERS*/ void
__curand_veccopy(unsigned int *vector, unsigned int *vectorA, int n)
{
  for(int i = 0; i < n; i++)
  {
    vector[i] = vectorA[i];
  }
}

/* copy matrixA to matrix */
/*QUALIFIERS*/ void
__curand_matcopy(unsigned int *matrix, unsigned int *matrixA, int n)
{
  for(int i = 0; i < n * n * 32; i++)
  {
    matrix[i] = matrixA[i];
  }
}

/* compute matrixA to power p, store result in matrix */
/*QUALIFIERS*/ void __curand_matpow(
  unsigned int *matrix,
  unsigned int *matrixA,
  unsigned long long p,
  int n)
{
  unsigned int matrixR[MAX_XOR_N * MAX_XOR_N * 32];
  unsigned int matrixS[MAX_XOR_N * MAX_XOR_N * 32];
  __curand_matidentity(matrix, n);
  __curand_matcopy(matrixR, matrixA, n);
  while(p)
  {
    if(p & 1)
    {
      __curand_matmat(matrix, matrixR, n);
    }
    __curand_matcopy(matrixS, matrixR, n);
    __curand_matmat(matrixR, matrixS, n);
    p >>= 1;
  }
}

/* Convert unsigned int to float, use no intrinsics */
/*QUALIFIERS*/ float __curand_uint32AsFloat(unsigned int i)
{
  union _xx
  {
    float f;
    unsigned int i;
  } xx;
  xx.i = i;
  return xx.f;
}

/* Convert two unsigned ints to double, use no intrinsics */
/*QUALIFIERS*/ double
__curand_hilouint32AsDouble(unsigned int hi, unsigned int lo)
{
  union _xx
  {
    double f;
    unsigned int hi;
    unsigned int lo;
  } xx;
  xx.hi = hi;
  xx.lo = lo;
  return xx.f;
}

/* Convert unsigned int to float, as efficiently as possible */
/*QUALIFIERS*/ float __curand_uint32_as_float(unsigned int x)
{
#if __CUDA_ARCH__ > 0
  return __int_as_float(x);
#elif !defined(__CUDA_ARCH__)
  return __curand_uint32AsFloat(x);
#endif
}

/*
QUALIFIERS double __curand_hilouint32_as_double(unsigned int hi, unsigned int lo)
{
#if __CUDA_ARCH__ > 0
    return __hiloint2double(hi, lo);
#elif !defined(__CUDA_ARCH__)
    return hilouint32AsDouble(hi, lo);
#endif
}
*/

/****************************************************************************/
/* Kernel implementations of RNGs */
/****************************************************************************/

/* Test RNG */

/*QUALIFIERS*/ void curand_init(
  unsigned long long seed,
  unsigned long long subsequence,
  unsigned long long offset,
  curandStateTest_t *state)
{
  state->v = (unsigned int)(seed * 3) + (unsigned int)(subsequence * 31337) +
             (unsigned int)offset;
}

/*QUALIFIERS*/ unsigned int curand(curandStateTest_t *state)
{
  unsigned int r = state->v++;
  return r;
}

/*QUALIFIERS*/ void skipahead(unsigned long long n, curandStateTest_t *state)
{
  state->v += (unsigned int)n;
}

/* XORWOW RNG */

template <typename T, int n>
/*QUALIFIERS*/ void
__curand_generate_skipahead_matrix_xor(unsigned int matrix[])
{
  T state;
  // Generate matrix that advances one step
  // matrix has n * n * 32 32-bit elements
  // solve for matrix by stepping single bit states
  for(int i = 0; i < 32 * n; i++)
  {
    state.d = 0;
    for(int j = 0; j < n; j++)
    {
      state.v[j] = 0;
    }
    state.v[i / 32] = (1 << (i & 31));
    curand(&state);
    for(int j = 0; j < n; j++)
    {
      matrix[i * n + j] = state.v[j];
    }
  }
}

template <typename T, int n>
/*QUALIFIERS*/ void
_skipahead_scratch(unsigned long long x, T *state, unsigned int *scratch)
{
  // unsigned int matrix[n * n * 32];
  unsigned int *matrix = scratch;
  // unsigned int matrixA[n * n * 32];
  unsigned int *matrixA = scratch + (n * n * 32);
  // unsigned int vector[n];
  unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
  // unsigned int result[n];
  unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
  unsigned long long p = x;
  for(int i = 0; i < n; i++)
  {
    vector[i] = state->v[i];
  }
  int matrix_num = 0;
  while(p && matrix_num < PRECALC_NUM_MATRICES - 1)
  {
    for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++)
    {
#ifdef __CUDA_ARCH__
      __curand_matvec(
        vector, precalc_xorwow_offset_matrix[matrix_num], result, n);
#else
//            __curand_matvec(vector, precalc_xorwow_offset_matrix_host[matrix_num], result, n);
#endif
      __curand_veccopy(vector, result, n);
    }
    p >>= PRECALC_BLOCK_SIZE;
    matrix_num++;
  }
  if(p)
  {
#ifdef __CUDA_ARCH__
    __curand_matcopy(
      matrix, precalc_xorwow_offset_matrix[PRECALC_NUM_MATRICES - 1], n);
    __curand_matcopy(
      matrixA, precalc_xorwow_offset_matrix[PRECALC_NUM_MATRICES - 1], n);
#else
//        __curand_matcopy(matrix, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
//        __curand_matcopy(matrixA, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
#endif
  }
  while(p)
  {
    for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++)
    {
      __curand_matvec(vector, matrixA, result, n);
      __curand_veccopy(vector, result, n);
    }
    p >>= SKIPAHEAD_BLOCKSIZE;
    if(p)
    {
      for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++)
      {
        __curand_matmat(matrix, matrixA, n);
        __curand_matcopy(matrixA, matrix, n);
      }
    }
  }
  for(int i = 0; i < n; i++)
  {
    state->v[i] = vector[i];
  }
  state->d += 362437 * (unsigned int)x;
}

template <typename T, int n>
/*QUALIFIERS*/ void _skipahead_sequence_scratch(
  unsigned long long x,
  T *state,
  unsigned int *scratch)
{
  // unsigned int matrix[n * n * 32];
  unsigned int *matrix = scratch;
  // unsigned int matrixA[n * n * 32];
  unsigned int *matrixA = scratch + (n * n * 32);
  // unsigned int vector[n];
  unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
  // unsigned int result[n];
  unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
  unsigned long long p = x;
  for(int i = 0; i < n; i++)
  {
    vector[i] = state->v[i];
  }
  int matrix_num = 0;
  while(p && matrix_num < PRECALC_NUM_MATRICES - 1)
  {
    for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++)
    {
#ifdef __CUDA_ARCH__
      __curand_matvec(vector, precalc_xorwow_matrix[matrix_num], result, n);
#else
//            __curand_matvec(vector, precalc_xorwow_matrix_host[matrix_num], result, n);
#endif
      __curand_veccopy(vector, result, n);
    }
    p >>= PRECALC_BLOCK_SIZE;
    matrix_num++;
  }
  if(p)
  {
#ifdef __CUDA_ARCH__
    __curand_matcopy(
      matrix, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
    __curand_matcopy(
      matrixA, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
#else
//        __curand_matcopy(matrix, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
//        __curand_matcopy(matrixA, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
#endif
  }
  while(p)
  {
    for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++)
    {
      __curand_matvec(vector, matrixA, result, n);
      __curand_veccopy(vector, result, n);
    }
    p >>= SKIPAHEAD_BLOCKSIZE;
    if(p)
    {
      for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++)
      {
        __curand_matmat(matrix, matrixA, n);
        __curand_matcopy(matrixA, matrix, n);
      }
    }
  }
  for(int i = 0; i < n; i++)
  {
    state->v[i] = vector[i];
  }
  /* No update of state->d needed, guaranteed to be a multiple of 2^32 */
}

/**
 * \brief Update XORWOW state to skip \p n elements.
 *
 * Update the XORWOW state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
/*QUALIFIERS*/ void skipahead(unsigned long long n, curandStateXORWOW_t *state)
{
  unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
  _skipahead_scratch<curandStateXORWOW_t, 5>(n, state, (unsigned int *)scratch);
}

/**
 * \brief Update XORWOW state to skip ahead \p n subsequences.
 *
 * Update the XORWOW state in \p state to skip ahead \p n subsequences.  Each
 * subsequence is \f$ 2^{67} \f$ elements long, so this means the function will skip ahead
 * \f$ 2^{67} \cdot n\f$ elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of subsequences to skip
 * \param state - Pointer to state to update
 */
/*QUALIFIERS*/ void
skipahead_sequence(unsigned long long n, curandStateXORWOW_t *state)
{
  unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
  _skipahead_sequence_scratch<curandStateXORWOW_t, 5>(
    n, state, (unsigned int *)scratch);
}

/*QUALIFIERS*/ void _curand_init_scratch(
  unsigned long long seed,
  unsigned long long subsequence,
  unsigned long long offset,
  curandStateXORWOW_t *state,
  unsigned int *scratch)
{
  // Break up seed, apply salt
  // Constants are arbitrary nonzero values
  unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
  unsigned int s1;
  s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
  // Simple multiplication to mix up bits
  // Constants are arbitrary odd values
  unsigned int t0 = 1099087573UL * s0;
  unsigned int t1 = 2591861531UL * s1;
  state->d = 6615241 + t1 + t0;
  state->v[0] = 123456789UL + t0;
  state->v[1] = 362436069UL ^ t0;
  state->v[2] = 521288629UL + t1;
  state->v[3] = 88675123UL ^ t1;
  state->v[4] = 5783321UL + t0;
  _skipahead_scratch<curandStateXORWOW_t, 5>(offset, state, scratch);
  _skipahead_sequence_scratch<curandStateXORWOW_t, 5>(
    subsequence, state, scratch);
  state->boxmuller_flag = 0;
}

/**
 * \brief Initialize XORWOW state.
 *
 * Initialize XORWOW state in \p state with the given \p seed, \p subsequence,
 * and \p offset.
 *
 * All input values of \p seed, \p subsequence, and \p offset are legal.  Large
 * values for \p subsequence and \p offset require more computation and so will
 * take more time to complete.
 *
 * A value of 0 for \p seed sets the state to the values of the original
 * published version of the \p xorwow algorithm.
 *
 * \param seed - Arbitrary bits to use as a seed
 * \param subsequence - Subsequence to start at
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
/*QUALIFIERS*/ void curand_init(
  unsigned long long seed,
  unsigned long long subsequence,
  unsigned long long offset,
  curandStateXORWOW_t *state)
{
  unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
  _curand_init_scratch(
    seed, subsequence, offset, state, (unsigned int *)scratch);
}
/**
 * \brief Return 32-bits of pseudorandomness from an XORWOW generator.
 *
 * Return 32-bits of pseudorandomness from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
 */
/*QUALIFIERS*/ unsigned int curand(curandStateXORWOW_t *state)
{
  unsigned int t;
  t = (state->v[0] ^ (state->v[0] >> 2));
  state->v[0] = state->v[1];
  state->v[1] = state->v[2];
  state->v[2] = state->v[3];
  state->v[3] = state->v[4];
  state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
  state->d += 362437;
  return state->v[4] + state->d;
}

//#############

/**
 * \brief Update Sobol32 state to skip \p n elements.
 *
 * Update the Sobol32 state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
template <typename T>
/*QUALIFIERS*/ void skipahead(unsigned int n, T state)
{
  unsigned int i_gray;
  state->i += n;
  /* Convert state->i to gray code */
  i_gray = state->i ^ (state->i >> 1);
  for(unsigned int k = 0; k < 32; k++)
  {
    if(i_gray & (1 << k))
    {
      state->x ^= state->direction_vectors[k];
    }
  }
  return;
}

/**
 * \brief Update Sobol64 state to skip \p n elements.
 *
 * Update the Sobol64 state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
template <typename T>
/*QUALIFIERS*/ void skipahead(unsigned long long n, T state)
{
  unsigned long long i_gray;
  state->i += n;
  /* Convert state->i to gray code */
  i_gray = state->i ^ (state->i >> 1);
  for(unsigned k = 0; k < 64; k++)
  {
    if(i_gray & (1ULL << k))
    {
      state->x ^= state->direction_vectors[k];
    }
  }
  return;
}

/**
 * \brief Initialize Sobol32 state.
 *
 * Initialize Sobol32 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 32 unsigned ints.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
 * direction vectors for the desired dimension
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
/*in curand.h*/

/*QUALIFIERS void curand_init(curandDirectionVectors32_t direction_vectors,
                                            unsigned int offset,
                                            curandStateSobol32_t *state)
{
    state->i = 0;
    for(int i = 0; i < 32; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = 0;
    skipahead<curandStateSobol32_t *>(offset, state);
}*/

/**
 * \brief Initialize Scrambled Sobol32 state.
 *
 * Initialize Sobol32 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 32 unsigned ints.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
 direction vectors for the desired dimension
 * \param scramble_c Scramble constant
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */

/*in curand.h*/
/*QUALIFIERS void curand_init(curandDirectionVectors32_t direction_vectors,
                                            unsigned int scramble_c,
                                            unsigned int offset,
                                            curandStateScrambledSobol32_t *state)
{
    state->i = 0;
    for(int i = 0; i < 32; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = scramble_c;
    skipahead<curandStateScrambledSobol32_t *>(offset, state);
}
*/
template <typename XT>
/*QUALIFIERS*/ int __curand_find_trailing_zero(XT x)
{
#if __CUDA_ARCH__ > 0
  unsigned long long z = x;
  int y = __ffsll(~z) | 0x40;
  return (y - 1) & 0x3F;
#else
  unsigned long long z = x;
  int i = 1;
  while(z & 1)
  {
    i++;
    z >>= 1;
  }
  return i - 1;
#endif
}
/**
 * \brief Initialize Sobol64 state.
 *
 * Initialize Sobol64 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 64 unsigned long longs.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
 direction vectors for the desired dimension
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */

/*in curand.h*/
/*QUALIFIERS*/ void curand_init(
  curandDirectionVectors64_t direction_vectors,
  unsigned long long offset,
  curandStateSobol64_t *state)
{
  state->i = 0;
  for(int i = 0; i < 64; i++)
  {
    state->direction_vectors[i] = direction_vectors[i];
  }
  state->x = 0;
  skipahead<curandStateSobol64_t *>(offset, state);
}

template <typename PT>
/*QUALIFIERS*/ void _skipahead_stride(int n_log2, PT state)
{
  /* Moving from i to i+2^n_log2 element in gray code is flipping two bits */
  unsigned int shifted_i = state->i >> n_log2;
  state->x ^= state->direction_vectors[n_log2 - 1];
  state->x ^=
    state->direction_vectors[__curand_find_trailing_zero(shifted_i) + n_log2];
  state->i += 1 << n_log2;
}
/**
 * \brief Initialize Scrambled Sobol64 state.
 *
 * Initialize Sobol64 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 64 unsigned long longs.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
 direction vectors for the desired dimension
 * \param scramble_c Scramble constant
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
/*in curand.h*/
/*QUALIFIERS*/ void curand_init(
  curandDirectionVectors64_t direction_vectors,
  unsigned long long scramble_c,
  unsigned long long offset,
  curandStateScrambledSobol64_t *state)
{
  state->i = 0;
  for(int i = 0; i < 64; i++)
  {
    state->direction_vectors[i] = direction_vectors[i];
  }
  state->x = scramble_c;
  skipahead<curandStateScrambledSobol64_t *>(offset, state);
}

/**
 * \brief Return 32-bits of quasirandomness from a Sobol32 generator.
 *
 * Return 32-bits of quasirandomness from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
 */

/*QUALIFIERS*/ unsigned int curand(curandStateSobol32_t *state)
{
  /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
  unsigned int res = state->x;
  state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
  state->i++;
  return res;
}

/**
 * \brief Return 32-bits of quasirandomness from a scrambled Sobol32 generator.
 *
 * Return 32-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
 */

/*QUALIFIERS*/ unsigned int curand(curandStateScrambledSobol32_t *state)
{
  /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
  unsigned int res = state->x;
  state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
  state->i++;
  return res;
}

/**
 * \brief Return 64-bits of quasirandomness from a Sobol64 generator.
 *
 * Return 64-bits of quasirandomness from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
 */

/*QUALIFIERS*/ unsigned long long curand(curandStateSobol64_t *state)
{
  /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
  unsigned long long res = state->x;
  state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
  state->i++;
  return res;
}

/**
 * \brief Return 64-bits of quasirandomness from a scrambled Sobol64 generator.
 *
 * Return 64-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
 */

/*QUALIFIERS*/ unsigned long long curand(curandStateScrambledSobol64_t *state)
{
  /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
  unsigned long long res = state->x;
  state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
  state->i++;
  return res;
}

/******************************************************/

/*QUALIFIERS*/ float _curand_uniform(unsigned int x)
{
  return x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
}

/*QUALIFIERS*/ float _curand_uniform(unsigned long long x)
{
  unsigned int t;
  t = (unsigned int)(x >> 32);
  return t * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
}

/*QUALIFIERS*/ double _curand_uniform_double(unsigned int x)
{
  return x * CURAND_2POW32_INV_DOUBLE + (CURAND_2POW32_INV_DOUBLE / 2.0);
}

/*QUALIFIERS*/ double _curand_uniform_double(unsigned long long x)
{
  return (x >> 11) * CURAND_2POW53_INV_DOUBLE +
         (CURAND_2POW53_INV_DOUBLE / 2.0);
}

/*QUALIFIERS*/ double _curand_uniform_double_hq(unsigned int x, unsigned int y)
{
  unsigned long long z =
    (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  return z * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE / 2.0);
}

/*QUALIFIERS*/ double _curand_uniform_double_64(unsigned long long x)
{
  unsigned long long z;
  z = (unsigned long long)x >> (64 - 53);
  return z * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE / 2.0);
}

/*QUALIFIERS*/ float curand_uniform(curandStateTest_t *state)
{
  return _curand_uniform(curand(state));
}

/*QUALIFIERS*/ double curand_uniform_double(curandStateTest_t *state)
{
  return _curand_uniform_double(curand(state));
}

/**
 * \brief Return a uniformly distributed float from an XORWOW generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p curand() to
 * get enough random bits to create the return value.  The current
 * implementation uses one call.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
/*QUALIFIERS*/ float curand_uniform(curandStateXORWOW_t *state)
{
  return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from an XORWOW generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p curand() to
 * get enough random bits to create the return value.  The current
 * implementation uses exactly two calls.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
/*QUALIFIERS*/ double curand_uniform_double(curandStateXORWOW_t *state)
{
  unsigned int x, y;
  x = curand(state);
  y = curand(state);
  return _curand_uniform_double_hq(x, y);
}

/**
 * \brief Return a uniformly distributed float from a Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
/*QUALIFIERS*/ float curand_uniform(curandStateSobol32_t *state)
{
  return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
/*QUALIFIERS*/ double curand_uniform_double(curandStateSobol32_t *state)
{
  return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
/*QUALIFIERS*/ float curand_uniform(curandStateScrambledSobol32_t *state)
{
  return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
/*QUALIFIERS*/ double
curand_uniform_double(curandStateScrambledSobol32_t *state)
{
  return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
/*QUALIFIERS*/ float curand_uniform(curandStateSobol64_t *state)
{
  return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
/*QUALIFIERS*/ double curand_uniform_double(curandStateSobol64_t *state)
{
  return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
/*QUALIFIERS*/ float curand_uniform(curandStateScrambledSobol64_t *state)
{
  return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
/*QUALIFIERS*/ double
curand_uniform_double(curandStateScrambledSobol64_t *state)
{
  return _curand_uniform_double(curand(state));
}

#endif // !defined(CURAND_KERNEL_H_)
