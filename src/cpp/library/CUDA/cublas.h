#if !defined(CUBLAS_H_)
#  define CUBLAS_H_

typedef enum cublasstatus
{
  CUBLAS_STATUS_SUCCESS,
  CUBLAS_STATUS_NOT_INITIALIZED,
  CUBLAS_STATUS_ALLOC_FAILED,
  CUBLAS_STATUS_INVALID_VALUE,
  CUBLAS_STATUS_ARCH_MISMATCH,
  CUBLAS_STATUS_MAPPING_ERROR,
  CUBLAS_STATUS_EXECUTION_FAILED,
  CUBLAS_STATUS_INTERNAL_ERROR,
  CUBLAS_STATUS_NOT_SUPPORTED,
  CUBLAS_STATUS_LICENSE_ERROR
} custatusResult;

typedef enum cublasstatus cublasStatus_t;
typedef struct cublashandle
{
} cublasHandle_t;

typedef enum cublasoperation
{
  CUBLAS_OP_N,
  CUBLAS_OP_T,
  CUBLAS_OP_C
} cuoperation;

typedef enum cublasoperation cublasOperation_t;

/*
This function initializes the CUBLAS library and creates a handle to an opaque structure 
holding the CUBLAS library context. 
*/
cublasStatus_t cublasCreate(cublasHandle_t *handle)
{
  return CUBLAS_STATUS_SUCCESS;
}

/*
This function releases hardware resources used by the CUBLAS library.  
*/
cublasStatus_t cublasDestroy(cublasHandle_t handle)
{
  return CUBLAS_STATUS_SUCCESS;
}

/*	This function copies a tile of rows x cols elements from a matrix A in host 
memory space to a matrix B in GPU memory space. It is assumed that each element 
requires storage of elemSize bytes and that both matrices are stored in column-major
 format, with the leading dimension of the source matrix A and destination matrix B 
given in lda and ldb, respectively. The leading dimension indicates the number of rows 
of the allocated matrix, even if only a submatrix of it is being used. In general,
 B is a device pointer that points to an object, or part of an object, that was 
allocated in GPU memory space via cublasAlloc().
*/
cublasStatus_t cublasSetMatrix(
  int rows,
  int cols,
  int elemSize,
  const void *A,
  int lda,
  void *B,
  int ldb)
{
  //Due to the Fortran column major the ldb must be the rows of matrix A
  __ESBMC_assert(ldb == rows, "Full matrix is not bein copied");

  return CUBLAS_STATUS_SUCCESS;
}

/*
This function copies a tile of rows x cols elements from a matrix A in GPU memory space
 to a matrix B in host memory space. It is assumed that each element requires storage 
of elemSize bytes and that both matrices are stored in column-major format, with the 
leading dimension of the source matrix A and destination matrix B given in lda and ldb,
 respectively. The leading dimension indicates the number of rows of the allocated
 matrix, even if only a submatrix of it is being used. In general, A is a device 
pointer that points to an object, or part of an object, that was allocated in GPU 
memory space via cublasAlloc(). 
*/
cublasStatus_t cublasGetMatrix(
  int rows,
  int cols,
  int elemSize,
  const void *A,
  int lda,
  void *B,
  int ldb)
{
  __ESBMC_assert(lda == cols, "Full matrix is not bein recovered");
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasSetVector(int n, int elemSize, const void *A, int lda, void *B, int ldb)
{
  __ESBMC_assert(lda == ldb, "Full matrix is not bein copied");

  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasGetVector(int n, int elemSize, const void *A, int lda, void *B, int ldb)
{
  __ESBMC_assert(ldb == lda, "Full matrix is not bein copied");

  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(
  cublasHandle_t handle,
  int n,
  const float *alpha,
  float *x,
  int incx)
{
  for (int i = 0; i < n; i++)
  {
    int j = 1 + (i - 1) * incx;
    x[j] = ((float)alpha[0]) * x[j];
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal(
  cublasHandle_t handle,
  int n,
  const double *alpha,
  double *x,
  int incx)
{
  for (int i = 0; i < n; i++)
  {
    int j = 1 + (i - 1) * incx;
    x[j] = ((double)alpha[0]) * x[j];
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSswap(
  cublasHandle_t handle,
  int n,
  float *x,
  int incx,
  float *y,
  int incy)
{
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    y[j] = x[k];
    x[k] = y[j];
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDswap(
  cublasHandle_t handle,
  int n,
  double *x,
  int incx,
  double *y,
  int incy)
{
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    y[j] = x[k];
    x[k] = y[j];
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(
  cublasHandle_t handle,
  int n,
  const float *x,
  int incx,
  const float *y,
  int incy,
  float *result)
{
  float aux = 0;
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    aux = y[j] * x[k] + aux;
  }
  result[0] = aux;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(
  cublasHandle_t handle,
  int n,
  const double *x,
  int incx,
  const double *y,
  int incy,
  double *result)
{
  double aux = 0;
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    aux = y[j] * x[k] + aux;
  }
  result[0] = aux;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScopy(
  cublasHandle_t handle,
  int n,
  const float *x,
  int incx,
  float *y,
  int incy)
{
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    y[j] = x[k];
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDcopy(
  cublasHandle_t handle,
  int n,
  const double *x,
  int incx,
  double *y,
  int incy)
{
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    y[j] = x[k];
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm(
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m,
  int n,
  int k,
  const float *alpha,
  const float *A,
  int lda,
  const float *B,
  int ldb,
  const float *beta,
  float *C,
  int ldc)
{
  if ((transa == CUBLAS_OP_N) && (transb == CUBLAS_OP_N))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX + contadorZ * k] * B[contadorX * n + contadorY]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }

  else if ((transa == CUBLAS_OP_N) && (transb == CUBLAS_OP_T))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX + contadorZ * k] * B[contadorX + contadorY * n]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }
  else if ((transa == CUBLAS_OP_T) && (transb == CUBLAS_OP_N))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX * k + contadorZ] * B[contadorX * n + contadorY]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }
  else if ((transa == CUBLAS_OP_T) && (transb == CUBLAS_OP_T))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX * k + contadorZ] * B[contadorX + contadorY * n]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m,
  int n,
  int k,
  const double *alpha,
  const double *A,
  int lda,
  const double *B,
  int ldb,
  const double *beta,
  double *C,
  int ldc)
{
  if ((transa == CUBLAS_OP_N) && (transb == CUBLAS_OP_N))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX + contadorZ * k] * B[contadorX * n + contadorY]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }
  else if ((transa == CUBLAS_OP_N) && (transb == CUBLAS_OP_T))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX + contadorZ * k] * B[contadorX + contadorY * n]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }
  else if ((transa == CUBLAS_OP_T) && (transb == CUBLAS_OP_N))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX * k + contadorZ] * B[contadorX * n + contadorY]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }
  else if ((transa == CUBLAS_OP_T) && (transb == CUBLAS_OP_T))
  {
    for (int contadorZ = 0; contadorZ < m; contadorZ++)
    {
      for (int contadorY = 0; contadorY < n; contadorY++)
      {
        float result = 0;
        for (int contadorX = 0; contadorX < k; contadorX++)
        {
          result =
            (A[contadorX * k + contadorZ] * B[contadorX + contadorY * n]) +
            result;
        }
        C[contadorY + contadorZ * m] =
          alpha[0] * result + beta[0] * C[contadorY + contadorZ * m];
      }
    }
  }
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSaxpy(
  cublasHandle_t handle,
  int n,
  const float *alpha,
  const float *x,
  int incx,
  float *y,
  int incy)
{
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    y[j] = ((float)alpha[0]) * x[k] + y[j];
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(
  cublasHandle_t handle,
  int n,
  const double *alpha,
  const double *x,
  int incx,
  double *y,
  int incy)
{
  for (int i = 0; i < n; i++)
  {
    int k = 1 + (i - 1) * incx;
    int j = 1 + (i - 1) * incy;
    y[j] = ((double)alpha[0]) * x[k] + y[j];
  }
  return CUBLAS_STATUS_SUCCESS;
}

#endif /*CUBLAS_H_*/
