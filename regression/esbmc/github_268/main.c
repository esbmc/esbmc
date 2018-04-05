#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <complex.h>
#include <string.h>
//#include <malloc.h>

/* miscellaneous constants */
#define	VNULL ((VEC *)NULL)
#define	MNULL ((MAT *)NULL)

/* macros that also check types and sets pointers to NULL */
#define	M_FREE(mat) (m_free(mat), (mat)=(MAT *)NULL)
#define V_FREE(vec)	(v_free(vec), (vec)=(VEC *)NULL)

#define MACHEPS 2.22045e-16

#define	m_output(mat) m_foutput(stdout, mat)

static const char *format = "%14.9g ";

#ifndef ANSI_C
#define ANSI_C 1
#endif

#define SEGMENTED

#ifndef THREADSAFE /* for use as a shared library */
#define THREADSAFE 1
#endif

#define TYPE_MAT mem_bytes(0, 0, sizeof(MAT))
#define TYPE_VEC mem_bytes(0, 0, sizeof(VEC))

#define	v_chk_idx(x, i) ((i)>=0 && (i)<(x)->dim)

#define v_get_val(x, i) (v_chk_idx(x, i) ? (x)->ve[(i)] : \
    (printf("Error!\n")))

#define	v_entry(x, i) v_get_val(x, i)

#define	v_set_val(x, i, val) (x->ve[i] = val)

#define	m_set_val(A, i, j, val) ((A)->me[(i)][(j)] = (val))

#define	m_add_val(A, i, j, val)	((A)->me[(i)][(j)] += (val))

#define	m_chk_idx(A, i, j) ((i)>=0 && (i)<(A)->m && (j)>=0 && (j)<=(A)->n)

#define	m_get_val(A, i, j) (m_chk_idx(A, i, j) ? \
    (A)->me[(i)][(j)] : (printf("Error!")))

#define	m_entry(A, i, j) m_get_val(A, i, j)

#define printfc(c) printf("%f%c%fi\n", c.real, (c.imag>=0.0f)? '+':'\0', c.imag)

/* standard copy & zero functions */
#define	MEM_COPY(from, to, size) memmove((to), (from), (size))
#define	MEM_ZERO(where, size) memset((where), '\0', (size))

/* allocate one object of given type */
#define	NEW(type) ((type *)calloc((size_t)1, (size_t)sizeof(type)))

/* allocate num objects of given type */
#define	NEW_A(num, type)  ((type *)calloc((size_t)(num), (size_t)sizeof(type)))

#define	MEMCOPY(from, to, n_items, type) \
  MEM_COPY((char *)(from), (char *)(to), (unsigned)(n_items)*sizeof(type))

/* type independent min and max operations */
#ifndef max
#define	max(a, b) ((a) > (b) ? (a) : (b))
#endif /* max */
#ifndef min
#define	min(a, b) ((a) > (b) ? (b) : (a))
#endif /* min */

#ifndef THREADSAFE
#define MEM_STAT_REG(var, type) mem_stat_reg_list((void **)&(var),
                     type, 0, __FILE__, __LINE__)
#else
#define MEM_STAT_REG(var, type)
#endif

/* matrix definition */
typedef	struct
{
  unsigned int m, n;
  unsigned int max_m, max_n, max_size;
  double **me, *base;   /* base is base of alloc'd mem */
}
MAT;

/* vector definition */
typedef	struct
{
  unsigned int dim, max_dim;
  double *ve;
}
VEC;

/* complex number definition */
typedef	struct
{
  double real, imag;
}
CMPLX;

/******************************Matrix Functions******************************/

/* m_add -- matrix addition -- may be in-situ */
#ifndef ANSI_C
MAT	*m_add(mat1, mat2, out)
MAT	*mat1, *mat2, *out;
#else
MAT	*m_add(const MAT *mat1, const MAT *mat2, MAT *out)
#endif
{
  unsigned int m, n, i, j;

  m = mat1->m;	n = mat1->n;
  for(i=0; i<m; i++ )
  {
    for(j = 0; j < n; j++)
	  out->me[i][j] = mat1->me[i][j]+mat2->me[i][j];
  }

	return (out);
}

/* m_sub -- matrix subtraction -- may be in-situ */
#ifndef ANSI_C
MAT	*m_sub(mat1, mat2, out)
MAT	*mat1, *mat2, *out;
#else
MAT	*m_sub(const MAT *mat1, const MAT *mat2, MAT *out)
#endif
{
  unsigned int m, n, i, j;

  m = mat1->m;	n = mat1->n;
  for(i=0; i<m; i++)
  {
    for(j=0; j<n; j++)
	  out->me[i][j] = mat1->me[i][j]-mat2->me[i][j];
  }

	return (out);
}

/* m_get -- gets an mxn matrix (in MAT form) by dynamic memory allocation
	-- normally ALL matrices should be obtained this way
	-- if either m or n is negative this will raise an error
	-- note that 0 x n and m x 0 matrices can be created */
MAT	*m_get(int m, int n)
{
  MAT *matrix = malloc(sizeof *matrix);
  int i, j;

  if(m < 0 || n < 0)
    printf("The matrix dimensions must be positive!\n");
  if((matrix=NEW(MAT)) == (MAT *)NULL)
    printf("The matrix is NULL!\n");

  matrix->m = m;        matrix->n = matrix->max_n = n;
  matrix->max_m = m;    matrix->max_size = m*n;

  matrix->me = (double **)malloc(m * sizeof(double*));
  for(int i = 0; i < m; i++)
    matrix->me[i] = (double *)malloc(n * sizeof(double));

  return (matrix);
}

/* m_resize -- returns the matrix A of size new_m x new_n; A is zeroed
   -- if A == NULL on entry then the effect is equivalent to m_get() */
MAT	*m_resize(MAT *A, int new_m, int new_n)
{
  int i;
  int new_max_m, new_max_n, old_m, old_n, add_rows;
  double **tmp;

  if(new_m < 0 || new_n < 0)
    printf("The size must be positive!");

  if(!A)
    return m_get(new_m, new_n);

  // nothing was changed
  if(new_m == A->m && new_n == A->n)
    return A;

  old_m = A->m;	old_n = A->n;    add_rows = new_m-old_m;
  if( new_m > A->max_m )
  {  // re-allocate A->me
  tmp = realloc(A->me, sizeof *A->me * (new_m));
    if(tmp)
    {
	  A->me = tmp;
      for(i = 0; i < add_rows; i++)
      {
        A->me[old_m + i] = malloc( sizeof *A->me[old_m + i] * old_n );
      }
    }
    if(new_n > A->max_n)
    {
      double *tmp;
      for(int i = 0; i < old_m; i++)
      {
        tmp = realloc(A->me[i], sizeof *A->me[i] * (new_n));
        if(tmp)
        {
          A->me[i] = tmp;
        }
      }
    }
    else if(new_n < A->max_n)
    {
      double *tmp;
        for(int i = 0; i < old_n; i++)
        {
          tmp = realloc(A->me[i], sizeof *A->me[i] * (new_n));
          if(tmp)
            A->me[i] = tmp;
        }
    }
  }
  else if(new_m < A->max_m)
  {
    int del_rows = old_m-new_m;
    double *tmp;
    for(int i = 1; i <= del_rows; i++)
    {
      free(A->me[old_m - i]);
      tmp = realloc(A->me, old_m - del_rows);
      if(tmp)
        A->me[i] = tmp;
    }
    if(new_n > A->max_n)
    {
      double *tmp;
      for(int i = 0; i < old_m; i++)
      {
        tmp = realloc( A->me[i], sizeof *A->me[i] * (new_n) );
        if(tmp)
        {
          A->me[i] = tmp;
        }
      }
    }
    else if(new_n < A->max_n)
    {
      double *tmp;
      for(int i = 0; i < old_n; i++)
      {
        tmp = realloc(A->me[i], sizeof *A->me[i] * (new_n));
        if(tmp)
          A->me[i] = tmp;
      }
    }
  }

  new_max_m = max(new_m, A->max_m);
  new_max_n = max(new_n, A->max_n);

  A->max_m = new_max_m;
  A->max_n = new_max_n;
  A->max_size = A->max_m*A->max_n;
  A->m = new_m; A->n = new_n;
  return A;
}

/* m_zero -- zero the matrix A */
#ifndef ANSI_C
MAT	*m_zero(A)
MAT *A;
#else
MAT	*m_zero(MAT *A)
#endif
{
  int i, j, A_m, A_n;
  double **A_me;

  A_m = A->m;	A_n = A->n;	A_me = A->me;
  for(i = 0; i < A_m; i++)
    for( j = 0; j < A_n; j++)
      A_me[i][j] = 0.0;
  return A;
}

/* __mltadd__ -- scalar multiply and add c.f. v_mltadd() */
#ifndef ANSI_C
void __mltadd__(dp1, dp2, s, len)
register double	*dp1, *dp2;
register double s;
register int len;
#else
void __mltadd__(double *dp1, const double *dp2, double s, int len)
#endif
{
  register int i;
#ifdef VUNROLL
  register int len4;

  len4 = len / 4;
  len  = len % 4;
  for(i = 0; i < len4; i++)
  {
    dp1[4*i]   += s*dp2[4*i];
    dp1[4*i+1] += s*dp2[4*i+1];
    dp1[4*i+2] += s*dp2[4*i+2];
    dp1[4*i+3] += s*dp2[4*i+3];
  }
  dp1 += 4*len4;    dp2 += 4*len4;
#endif

  for(i = 0; i < len; i++)
    dp1[i] += s*dp2[i];
}

/* m_mlt -- matrix-matrix multiplication */
#ifndef ANSI_C
MAT	*m_mlt(A, B, OUT)
MAT	*A, *B, *OUT;
#else
MAT	*m_mlt(const MAT *A, const MAT *B, MAT *OUT)
#endif
{
  unsigned int i, /* j, */ k, m, n, p;
  double	**A_v, **B_v, *B_row, *OUT_row, sum, tmp;

  m = A->m;	n = A->n; p = B->n;
  A_v = A->me; B_v = B->me;

  if(OUT==(MAT *)NULL || OUT->m != A->m || OUT->n != B->n)
    OUT = m_resize(OUT, A->m, B->n);
  m_zero(OUT);
  for(i=0; i<m; i++)
    for( k=0; k<n; k++)
    {
      if(A_v[i][k] != 0.0)
        __mltadd__(OUT->me[i], B_v[k], A_v[i][k], (int)p);
    }

  return OUT;
}

/* m_foutput -- prints a representation of the matrix a onto file/stream fp */
#ifndef ANSI_C
void m_foutput(fp, a)
FILE *fp;
MAT *a;
#else
void m_foutput(FILE *fp, const MAT *a)
#endif
{
  unsigned int i, j, tmp;

  if( a == (MAT *)NULL )
  {
    fprintf(fp, "Matrix: NULL\n");
    return;
  }
  fprintf(fp, "Matrix: %d by %d\n", a->m, a->n);
  if(a->me == (double **)NULL)
  {
    fprintf(fp, "NULL\n");
    return;
  }
  for(i = 0; i < a->m; i++)   /* for each row... */
  {
    fprintf(fp, "row %u: ", i);
    for(j = 0, tmp = 2; j < a->n; j++, tmp++)
    {             /* for each col in row... */
      fprintf(fp, format, a->me[i][j]);
        if(!(tmp % 5))
          putc('\n', fp);
    }
    if(tmp % 5 != 1)
      putc('\n', fp);
  }
}

/* m_copy -- copies matrix into new area
	-- out(i0:m,j0:n) <- in(i0:m,j0:n) */
#ifndef ANSI_C
MAT	*m_copy(in, out)
MAT	*in, *out;
#else
MAT	*m_copy(const MAT *in, MAT *out)
#endif
{
  unsigned int i0 = 0, j0 = 0;
  unsigned int i, j;

  if(in == out)
    return (out);
  if(out == MNULL || out->m < in->m || out->n < in->n )
    out = m_resize(out, in->m, in->n);

  for(i=i0; i < in->m; i++)
  {
    MEM_COPY(&(in->me[i][j0]), &(out->me[i][j0]),
    (in->n - j0)*sizeof(double));
  }
  return (out);
}

/* v_zero -- zero the vector x */
#ifndef ANSI_C
VEC	*v_zero(x)
VEC	*x;
#else
VEC	*v_zero(VEC *x)
#endif
{
  for(int i = 0; i < x->dim; i++)
    x->ve[i] = 0.0;
  return x;
}

/* v_get -- gets a VEC of dimension 'size'
   -- Note: initialized to zero */
#ifndef ANSI_C
VEC	*v_get(size)
int	size;
#else
VEC	*v_get(int size)
#endif
{
  VEC *vector = malloc(sizeof *vector);
  int i, j;

  vector->dim = vector->max_dim = size;
  if(size < 0)
    printf("The vector dimension must be positive!\n");
  if((vector->ve = NEW_A(size, double)) == (double *)NULL )
  {
    free(vector);
  }
  else
  {
    vector->ve = (double *)malloc(size * sizeof(double));
  }

  return (vector);
}

/* v_resize -- returns the vector x with dim new_dim
   -- x is set to the zero vector */
#ifndef ANSI_C
VEC	*v_resize(x, new_dim)
VEC	*x;
int	new_dim;
#else
VEC	*v_resize(VEC *x, int new_dim)
#endif
{
  double *ptr;
  if(!x)
    return v_get(new_dim);

  /* nothing is changed */
  if(new_dim == x->dim)
    return x;

  if( x->max_dim == 0 )	/* assume that it's from sub_vec */
    return v_get(new_dim);
  ptr = x->ve;
  if(new_dim > x->max_dim)
  {
    ptr = realloc(ptr, new_dim * sizeof *ptr);
  }
  if( new_dim > x->dim )
  {
    for(int i = 1; i < (new_dim - x->dim); i++)
      x->ve[new_dim-i] = 0.0;
  }
  else if(new_dim < x->dim)
  {
    ptr = realloc(ptr, new_dim * sizeof *ptr);
  }

  x->dim = new_dim;

  return x;
}

/* set_col -- sets column of matrix to values given in vec (in situ)
	-- that is, mat(i0:lim,col) <- vec(i0:lim) */
#ifndef ANSI_C
MAT	*set_col(mat, col, vec)
MAT	*mat;
VEC	*vec;
unsigned int col;
#else
MAT	*set_col(MAT *mat, unsigned int col, const VEC *vec/*, unsigned int i0*/)
#endif
{
  unsigned int i, lim, i0;

  lim = min(mat->m, vec->dim);
  for(i=i0; i<lim; i++)
    mat->me[i][col] = vec->ve[i];

  return (mat);
}

/* m_free -- returns MAT & associated memory back to memory heap */
#ifndef ANSI_C
int	m_free(mat)
MAT	*mat;
#else
int	m_free(MAT *mat)
#endif
{
#ifdef SEGMENTED
  int i;
#endif

  if(mat == (MAT *)NULL || (int)(mat->m) < 0 ||
     (int)(mat->n) < 0)
    return (-1);

#ifndef SEGMENTED
  if(mat->base != (double *)NULL)
  {
    mat->base = (double *)malloc(mat->max_m*mat->max_n * sizeof(double));
    free((char *)(mat->base));
  }
#else
  for( i = 0; i < mat->max_m; i++ )
    if(mat->me[i] != (double *)NULL)
    {
      mat->me[i] = (double *)malloc(mat->max_n * sizeof(double));
      free((char *)(mat->me[i]));
    }
#endif
  if(mat->me != (double **)NULL)
  {
    mat->me = (double **)malloc(mat->max_m * sizeof(double*));
    free((char *)(mat->me));
  }

  mat = malloc(sizeof *mat);
  free((char *)mat);

  return (0);
}

/* v_free -- returns VEC & associated memory back to memory heap */
#ifndef ANSI_C
int	v_free(vec)
VEC	*vec;
#else
int	v_free(VEC *vec)
#endif
{
  if( vec==(VEC *)NULL || (int)(vec->dim) < 0 )
    /* don't trust it */
    return (-1);

  if( vec->ve == (double *)NULL )
  {
    vec = malloc(sizeof *vec);
    free((char *)vec);
  }
  else
  {
    vec = malloc(sizeof *vec);
    vec->ve = (double *)malloc(vec->max_dim*sizeof(double));
    free((char *)vec->ve);
    free((char *)vec);
  }

  return (0);
}

/* v_copy -- copies vector into new area
	-- out(i0:dim) <- in(i0:dim) */
#ifndef ANSI_C
VEC	*v_copy(in, out)
VEC	*in, *out;
#else
VEC	*v_copy(const VEC *in, VEC *out)
#endif
{
  unsigned int i0 = 0;

  if(in == out)
    return (out);
  if(out == VNULL || out->dim < in->dim)
    out = v_resize(out, in->dim);

  MEM_COPY(&(in->ve[i0]), &(out->ve[i0]), (in->dim - i0)*sizeof(double));

  return (out);
}


/* __ip__ -- inner product */
#ifndef ANSI_C
double __ip__(dp1, dp2, len)
register double	*dp1, *dp2;
int len;
#else
  double __ip__(const double *dp1, const double *dp2, int len)
#endif
{
#ifdef VUNROLL
  register int len4;
  register double sum1, sum2, sum3;
#endif
  register int i;
  register double sum;

  sum = 0.0;
#ifdef VUNROLL
  sum1 = sum2 = sum3 = 0.0;

  len4 = len / 4;
  len  = len % 4;

  for(i = 0; i < len4; i++)
  {
    sum  += dp1[4*i]*dp2[4*i];
    sum1 += dp1[4*i+1]*dp2[4*i+1];
    sum2 += dp1[4*i+2]*dp2[4*i+2];
    sum3 += dp1[4*i+3]*dp2[4*i+3];
  }
  sum  += sum1 + sum2 + sum3;
  dp1 += 4*len4;	dp2 += 4*len4;
#endif
  for(i = 0; i < len; i++)
    sum  += dp1[i]*dp2[i];

  return sum;
}

/* m_inverse -- returns inverse of A, provided A is not too rank deficient
-- uses Gauss - Jordan */
#ifndef ANSI_C
MAT	*m_inverse(A, out)
MAT	*A, *out;
#else
MAT	*m_inverse(const MAT *A, MAT *out)
#endif
{
  int i, j, k, matsize;
  float temp;
  MAT *AUX = m_copy(A, MNULL);
  matsize = AUX->m;
  // automatically initialize the unit matrix, e.g.
  for(i = 0; i < matsize; i++)
  {
    for(j = 0; j < matsize; j++)
    {
      if(i == j)
      {
        out->me[i][j]=1;
      }
      else
        out->me[i][j]=0;
    }
  }
/*---------------Logic starts here------------------*/
  /* procedure to make the matrix A to unit matrix
   --by some row operations,and the same row operations of
   --Unit mat. I gives the inverse of matrix A
   --'temp' stores the A[k][k] value so that A[k][k] will not change
   --during the operation A[i][j]/=A[k][k] when i=j=k
  --*/
  for(k = 0; k < matsize; k++)
  {
    // it performs the following row operations to make A to unit matrix
    // R0=R0/A[0][0],similarly for I also R0=R0/A[0][0]
    // R1=R1-R0*A[1][0] similarly for I
    // R2=R2-R0*A[2][0]
    temp = AUX->me[k][k];
    for(j = 0; j < matsize; j++)
    {
      AUX->me[k][j] /= temp;
      out->me[k][j] /= temp;
    }
    for(i = 0; i < matsize; i++)
    {
      // R1=R1/A[1][1]
      // R0=R0-R1*A[0][1]
      // R2=R2-R1*A[2][1]
      temp = AUX->me[i][k];
      for(j = 0; j < matsize; j++)
      {
        if(i == k)
          break;
        // R2=R2/A[2][2]
        // R0=R0-R2*A[0][2]
        // R1=R1-R2*A[1][2]
        AUX->me[i][j] -= AUX->me[k][j]*temp;
        out->me[i][j] -= out->me[k][j]*temp;
      }
    }
  }
/*---------------Logic ends here--------------------*/

  return out;
}

/* mat_id -- set A to being closest to identity matrix as possible
  -- i.e. A[i][j] == 1 if i == j and 0 otherwise */
#ifndef ANSI_C
MAT	*m_ident(A)
MAT	*A;
#else
MAT	*m_ident(MAT *A)
#endif
{
  int i, size;
  m_zero(A);
  size = min(A->m, A->n);
  for(i = 0; i < size; i++)
    A->me[i][i] = 1.0;

  return A;
}

/* _m_pow -- computes integer powers of a square matrix A, A^p
   -- uses tmp as temporary workspace */
#ifndef ANSI_C
MAT	*_m_pow(A, p, tmp, out)
MAT	*A, *tmp, *out;
int	p;
#else
MAT	*_m_pow(const MAT *A, int p, MAT *tmp, MAT *out)
#endif
{
  int it_cnt, k, max_bit;

#define	Z(k) (((k) & 1) ? tmp : out)

  out = m_resize(out, A->m, A->n);
  tmp = m_resize(tmp, A->m, A->n);

  if(p == 0)
    out = m_ident(out);
  else if(p > 0)
  {
    it_cnt = 1;
    for(max_bit = 0; ; max_bit++)
      if((p >> (max_bit+1)) == 0)
        break;
      tmp = m_copy(A, tmp);

      for(k = 0; k < max_bit; k++)
      {
        m_mlt(Z(it_cnt), Z(it_cnt), Z(it_cnt+1));
        it_cnt++;
        if(p & (1 << (max_bit-1)))
        {
          m_mlt(A, Z(it_cnt), Z(it_cnt+1));
          it_cnt++;
        }
        p <<= 1;
      }
      if(it_cnt & 1)
        out = m_copy(Z(it_cnt), out);
  }

  return out;

#undef Z
}

/* m_same_elements -- fills matrix with one's */
#ifndef ANSI_C
MAT	*m_same_elements(A, u)
MAT *A;
double u;
#else
MAT	*m_same_elements(MAT *A, double u)
#endif
{
  int i, j;
  for(i = 0; i < A->m; i++)
    for ( j = 0; j < A->n; j++ )
      A->me[i][j] = u;

  return A;
}

/* m_pow -- computes integer powers of a square matrix A, A^p */
#ifndef ANSI_C
MAT	*m_pow(A, p, out)
MAT	*A, *out;
int	p;
#else
MAT	*m_pow(const MAT *A, int p, MAT *out)
#endif
{
  static MAT *wkspace=MNULL, *tmp=MNULL;

  wkspace = m_resize(wkspace, A->m, A->n);
  MEM_STAT_REG(wkspace, TYPE_MAT);
  if(p < 0)
  {
    tmp = m_resize(tmp, A->m, A->n);
    MEM_STAT_REG(tmp, TYPE_MAT);
    tmp = m_inverse(A, tmp);
    out = _m_pow(tmp, -p, wkspace, out);
  }
  else
    out = _m_pow(A, p, wkspace, out);

#ifdef THREADSAFE
  M_FREE(wkspace);  M_FREE(tmp);
#endif

  return out;
}

/* get_col -- gets a specified column of a matrix and retruns it as a vector */
#ifndef ANSI_C
VEC	*get_col(mat, col, vec)
unsigned int col;
MAT	*mat;
VEC	*vec;
#else
VEC	*get_col(const MAT *mat, unsigned int col, VEC *vec)
#endif
{
  unsigned int i;

  if(vec == (VEC *)NULL || vec->dim < mat->m)
    vec = v_resize(vec, mat->m);

  for(i = 0; i < mat->m; i++)
    vec->ve[i] = mat->me[i][col];

  return (vec);
}

/* _v_copy -- copies vector into new area
	-- out(i0:dim) <- in(i0:dim) */
#ifndef ANSI_C
VEC	*_v_copy(in, out, i0)
VEC	*in, *out;
unsigned int i0;
#else
VEC	*_v_copy(const VEC *in, VEC *out, unsigned int i0)
#endif
{
  if(in == out)
    return (out);
  if(out==VNULL || out->dim < in->dim)
    out = v_resize(out, in->dim);

  MEM_COPY(&(in->ve[i0]), &(out->ve[i0]), (in->dim - i0)*sizeof(double));

  return (out);
}

/* _in_prod -- inner product of two vectors from i0 downwards
   -- that is, returns a(i0:dim)^T.b(i0:dim) */
#ifndef ANSI_C
double	_in_prod(a, b, i0)
VEC	*a, *b;
unsigned int i0;
#else
double _in_prod(const VEC *a, const VEC *b, unsigned int i0)
#endif
{
  unsigned int limit;

  return __ip__(&(a->ve[i0]), &(b->ve[i0]), (int)(limit-i0));
}

/* hhvec -- calulates Householder vector to eliminate all entries after the
   i0 entry of the vector vec. It is returned as out. May be in-situ */
#ifndef ANSI_C
VEC	*hhvec(vec, i0, beta, out, newval)
VEC	*vec, *out;
unsigned int i0;
double *beta, *newval;
#else
VEC	*hhvec(const VEC *vec, unsigned int i0, double *beta,
           VEC *out, double *newval)
#endif
{
  double norm, temp;

  out = _v_copy(vec, out, i0);
  temp = (double)_in_prod(out, out, i0);
  norm = sqrt(temp);
  if(norm <= 0.0)
  {
    *beta = 0.0;
    return (out);
  }
  *beta = 1.0/(norm * (norm+fabs(out->ve[i0])));
  if(out->ve[i0] > 0.0)
    *newval = -norm;
  else
    *newval = norm;
  out->ve[i0] -= *newval;

  return (out);
}

/* _hhtrcols -- transform a matrix by a Householder vector by columns
    starting at row i0 from column j0
    -- that is, M(i0:m,j0:n) <- (I-beta.hh(i0:m).hh(i0:m)^T)M(i0:m,j0:n)
    -- in-situ
    -- scratch vector w passed as argument
    -- raises error if w == NULL
*/
#ifndef ANSI_C
MAT	*_hhtrcols(M, i0, j0, hh, beta, w)
MAT	*M;
unsigned int i0, j0;
VEC	*hh;
double	beta;
VEC	*w;
#else
MAT	*_hhtrcols(MAT *M, unsigned int i0, unsigned int j0,
               const VEC *hh, double beta, VEC *w)
#endif
{
  int	i;

  if(beta == 0.0)
    return (M);

  if(w->dim < M->n)
    w = v_resize(w, M->n);
  v_zero(w);

  for(i = i0; i < M->m; i++)
    if(hh->ve[i] != 0.0)
      __mltadd__(&(w->ve[j0]), &(M->me[i][j0]), hh->ve[i],
                 (int)(M->n-j0));
  for(i = i0; i < M->m; i++)
    if(hh->ve[i] != 0.0)
      __mltadd__(&(M->me[i][j0]), &(w->ve[j0]), -beta*hh->ve[i],
                 (int)(M->n-j0));
  return (M);
}

/* hhtrrows -- transform a matrix by a Householder vector by rows
    starting at row i0 from column j0 -- in-situ
    -- that is, M(i0:m,j0:n) <- M(i0:m,j0:n)(I-beta.hh(j0:n).hh(j0:n)^T) */
#ifndef ANSI_C
MAT	*hhtrrows(M, i0, j0, hh, beta)
MAT	*M;
unsigned int i0, j0;
VEC	*hh;
double beta;
#else
MAT	*hhtrrows(MAT *M, unsigned int i0, unsigned int j0,
              const VEC *hh, double beta)
#endif
{
  double ip, scale;
  int i;

  if(beta == 0.0)
    return (M);

  /* for each row ... */
  for(i = i0; i < M->m; i++)
  { /* compute inner product */
    ip = __ip__(&(M->me[i][j0]), &(hh->ve[j0]), (int)(M->n-j0));
    scale = beta*ip;
    if(scale == 0.0)
      continue;

    /* do operation */
    __mltadd__(&(M->me[i][j0]), &(hh->ve[j0]), -scale,
               (int)(M->n-j0));
  }

  return (M);
}

/* Hfactor -- compute Hessenberg factorization in compact form.
    -- factorization performed in situ
*/
#ifndef ANSI_C
MAT	*Hfactor(A, diag, beta)
MAT	*A;
VEC	*diag, *beta;
#else
MAT	*Hfactor(MAT *A, VEC *diag, VEC *beta)
#endif
{
  static VEC *hh = VNULL, *w = VNULL;
  int k, limit;
  double b;

  limit = A->m - 1;

  hh = v_resize(hh, A->m);
  w  = v_resize(w, A->n);
  MEM_STAT_REG(hh, TYPE_VEC);
  MEM_STAT_REG(w, TYPE_VEC);

  for(k = 0; k < limit; k++)
  {
    /* compute the Householder vector hh */
    get_col(A, (unsigned int)k, hh);
    hhvec(hh, k+1, &beta->ve[k], hh, &A->me[k+1][k]);
    v_set_val(diag, k, v_entry(hh, k+1));

    /* apply Householder operation symmetrically to A */
    b = v_entry(beta, k);
    _hhtrcols(A, k+1, k+1, hh, b, w);
    hhtrrows(A, 0, k+1, hh, b);
  }

#ifdef THREADSAFE
  V_FREE(hh);  V_FREE(w);
#endif

  return (A);
}

/* hhtrvec -- apply Householder transformation to vector 
    -- that is, out <- (I-beta.hh(i0:n).hh(i0:n)^T).in
    -- may be in-situ */
#ifndef ANSI_C
VEC	*hhtrvec(hh, beta, i0, in, out)
VEC	*hh, *in, *out;	/* hh = Householder vector */
unsigned int i0;
double beta;
#else
VEC	*hhtrvec(const VEC *hh, double beta, unsigned int i0,
             const VEC *in, VEC *out)
#endif
{
  double scale, temp;
  temp = (double)_in_prod(hh, in, i0);
  scale = beta*temp;
  out = v_copy(in, out);
  __mltadd__(&(out->ve[i0]), &(hh->ve[i0]), -scale, (int)(in->dim-i0));

  return (out);
}

/* makeHQ -- construct the Hessenberg orthogonalising matrix Q;
    -- i.e. Hess M = Q.M.Q'	*/
#ifndef ANSI_C
MAT	*makeHQ(H, diag, beta, Qout)
MAT	*H, *Qout;
VEC	*diag, *beta;
#else
MAT	*makeHQ(MAT *H, VEC *diag, VEC *beta, MAT *Qout)
#endif
{
  int i, j, limit;
  static VEC *tmp1 = VNULL, *tmp2 = VNULL;

  Qout = m_resize(Qout, H->m, H->m);

  tmp1 = v_resize(tmp1, H->m);
  tmp2 = v_resize(tmp2, H->m);
  MEM_STAT_REG(tmp1, TYPE_VEC);
  MEM_STAT_REG(tmp2, TYPE_VEC);

  for(i = 0; i < H->m; i++)
  {
    /* tmp1 = i'th basis vector */
    for(j = 0; j < H->m; j++)
      v_set_val(tmp1, j, 0.0);
    v_set_val(tmp1, i, 1.0);

    /* apply H/h transforms in reverse order */
    for(j = limit-1; j >= 0; j--)
    {
      get_col(H, (unsigned int)j, tmp2);
      v_set_val(tmp2, j+1, v_entry(diag, j));
      hhtrvec(tmp2, beta->ve[j], j+1, tmp1, tmp1);
    }

    /* insert into Qout */
    set_col(Qout, (unsigned int)i, tmp1);
  }

#ifdef THREADSAFE
  V_FREE(tmp1);  V_FREE(tmp2);
#endif

  return (Qout);
}

/* makeH -- construct actual Hessenberg matrix */
#ifndef ANSI_C
MAT	*makeH(H, Hout)
MAT	*H, *Hout;
#else
MAT	*makeH(const MAT *H, MAT *Hout)
#endif
{
  int i, j, limit;

  Hout = m_resize(Hout, H->m, H->m);
  Hout = m_copy(H, Hout);

  limit = H->m;
  for(i = 1; i < limit; i++)
    for(j = 0; j < i-1; j++)
      m_set_val(Hout, i, j, 0.0);

  return (Hout);
}

/* rot_cols -- postmultiply mat by givens rotation described by c, s */
#ifndef ANSI_C
MAT	*rot_cols(mat, i, k, c, s, out)
MAT	*mat, *out;
unsigned int i, k;
double c, s;
#else
MAT	*rot_cols(const MAT *mat, unsigned int i, unsigned int k,
              double c, double s, MAT *out)
#endif
{
  unsigned int j;
  double temp;

  if(mat != out)
    out = m_copy(mat, m_resize(out, mat->m, mat->n));

  for(j=0; j<mat->m; j++)
  {
    temp = c*m_entry(out, j, i) + s*m_entry(out, j, k);
    m_set_val(out, j, k, -s*m_entry(out, j, i) + c*m_entry(out, j, k));
    m_set_val(out, j, i, temp);
  }

  return (out);
}

/* rot_rows -- premultiply mat by givens rotation described by c, s */
#ifndef ANSI_C
MAT	*rot_rows(mat, i, k, c, s, out)
MAT	*mat, *out;
unsigned int i, k;
double c, s;
#else
MAT	*rot_rows(const MAT *mat, unsigned int i, unsigned int k,
              double c, double s, MAT *out)
#endif
{
  unsigned int j;
  double temp;

  if(mat != out)
    out = m_copy(mat, m_resize(out, mat->m, mat->n));

  for(j=0; j<mat->n; j++)
  {
    temp = c*m_entry(out, i, j) + s*m_entry(out, k, j);
    m_set_val(out, k, j, -s*m_entry(out, i, j) + c*m_entry(out, k, j));
    m_set_val(out, i, j, temp);
  }

  return (out);
}

/* hhldr3 -- computes */
#ifndef ANSI_C
static void hhldr3(x, y, z, nu1, beta, newval)
double x, y, z;
double *nu1, *beta, *newval;
#else
static void hhldr3(double x, double y, double z,
                   double *nu1, double *beta, double *newval)
#endif
{
  double alpha;

  if(x >= 0.0)
    alpha = sqrt(x*x+y*y+z*z);
  else
    alpha = -sqrt(x*x+y*y+z*z);
  *nu1 = x + alpha;
  *beta = 1.0/(alpha*(*nu1));
  *newval = alpha;
}

/*hhldr3rows */
#ifndef ANSI_C
static void hhldr3rows(A, k, i0, beta, nu1, nu2, nu3)
MAT	*A;
int	k, i0;
double beta, nu1, nu2, nu3;
#else
static void hhldr3rows(MAT *A, int k, int i0, double beta,
                       double nu1, double nu2, double nu3)
#endif
{
  double **A_me, ip, prod;
  int i, m;
  A_me = A->me;  m = A->m;
  i0 = min(i0, m-1);

  for(i = 0; i <= i0; i++)
  {
    ip = nu1*m_entry(A, i, k) + nu2*m_entry(A, i, k+1)+nu3 * m_entry(A, i, k+2);
    prod = ip*beta;
    m_add_val(A, i, k, -prod*nu1);
    m_add_val(A, i, k+1, -prod*nu2);
    m_add_val(A, i, k+2, -prod*nu3);
  }
}

/* givens -- returns c,s parameters for Givens rotation to
       eliminate y in the vector [ x y ]' */
#ifndef ANSI_C
void givens(x, y, c, s)
double x, y;
double *c, *s;
#else
void givens(double x, double y, double *c, double *s)
#endif
{
  double norm;

  norm = sqrt(x*x+y*y);
  if(norm == 0.0)
  {
    *c = 1.0;
    *s = 0.0;
  }	/* identity */
  else
  {
    *c = x/norm;
    *s = y/norm;
  }
}

/* schur -- computes the Schur decomposition of the matrix A in situ
    -- optionally, gives Q matrix such that Q^T.A.Q is upper triangular
    -- returns upper triangular Schur matrix */
#ifndef ANSI_C
MAT	*schur(A, Q)
MAT	*A, *Q;
#else
MAT	*schur(MAT *A, MAT *Q)
#endif
{
  int i, j, iter, k, k_min, k_max, k_tmp, n, split;
  double beta2, c, discrim, dummy, nu1, s, t, tmp, x, y, z;
  double **A_me;
  double sqrt_macheps;
  static VEC *diag=VNULL, *beta=VNULL;

  n = A->n;
  diag = v_resize(diag, A->n);
  beta = v_resize(beta, A->n);
  MEM_STAT_REG(diag, TYPE_VEC);
  MEM_STAT_REG(beta, TYPE_VEC);
  /* compute Hessenberg form */
  Hfactor(A, diag, beta);

  /* save Q if necessary */
  if(Q)
    Q = makeHQ(A, diag, beta, Q);
  makeH(A, A);

  sqrt_macheps = sqrt(MACHEPS);

  k_min = 0;
  A_me = A->me;

  while(k_min < n)
  {
    double a00, a01, a10, a11;
    double scale, t, numer, denom;

    /* find k_max to suit:
       submatrix k_min..k_max should be irreducible */
    k_max = n-1;
    for(k = k_min; k < k_max; k++)
      if(m_entry(A, k+1, k) == 0.0)
      {
        k_max = k;
        break;
      }

    if(k_max <= k_min)
    {
      k_min = k_max + 1;
      continue;      /* outer loop */
    }

    /* check to see if we have a 2 x 2 block
       with complex eigenvalues */
    if(k_max == k_min + 1)
    {
      a00 = m_entry(A, k_min, k_min);
      a01 = m_entry(A, k_min, k_max);
      a10 = m_entry(A, k_max, k_min);
      a11 = m_entry(A, k_max, k_max);
      tmp = a00 - a11;
      discrim = tmp*tmp + 4*a01*a10;
      if(discrim < 0.0)
      {
        /* yes -- e-vals are complex
               -- put 2 x 2 block in form [a b; c a];
        then eigenvalues have real part a & imag part sqrt(|bc|) */
        numer = - tmp;
        denom = (a01+a10 >= 0.0) ?
                (a01+a10) + sqrt((a01+a10)*(a01+a10)+tmp*tmp) :
                (a01+a10) - sqrt((a01+a10)*(a01+a10)+tmp*tmp);
        if(denom != 0.0)
        {    /* t = s/c = numer/denom */
          t = numer/denom;
          scale = c = 1.0/sqrt(1+t*t);
          s = c*t;
        }
        else
        {
          c = 1.0;
          s = 0.0;
        }
        rot_cols(A, k_min, k_max, c, s, A);
        rot_rows(A, k_min, k_max, c, s, A);
        if(Q != MNULL)
          rot_cols(Q, k_min, k_max, c, s, Q);
        k_min = k_max + 1;
        continue;
      }
      else
      {
        /* discrim >= 0; i.e. block has two real eigenvalues */
        /* no -- e-vals are not complex;
         split 2 x 2 block and continue */
        /* s/c = numer/denom */
        numer = (tmp >= 0.0) ?
              - tmp - sqrt(discrim) : - tmp + sqrt(discrim);
        denom = 2*a01;
        if(fabs(numer) < fabs(denom))
        {    /* t = s/c = numer/denom */
          t = numer/denom;
          scale = c = 1.0/sqrt(1+t*t);
          s = c*t;
        }
        else if(numer != 0.0)
        {    /* t = c/s = denom/numer */
          t = denom/numer;
          scale = 1.0/sqrt(1+t*t);
          c = fabs(t)*scale;
          s = (t >= 0.0) ? scale : -scale;
        }
        else /* numer == denom == 0 */
        {
          c = 0.0;
          s = 1.0;
        }
        rot_cols(A, k_min, k_max, c, s, A);
        rot_rows(A, k_min, k_max, c, s, A);
        if(Q != MNULL)
          rot_cols(Q, k_min, k_max, c, s, Q);
        k_min = k_max + 1;  /* go to next block */
        continue;
      }
    }

    /* now have r x r block with r >= 2:
     apply Francis QR step until block splits */
    split = 0;
    iter = 0;
    while(!split)
    {
      iter++;
      /* set up Wilkinson/Francis complex shift */
      k_tmp = k_max - 1;

      a00 = m_entry(A, k_tmp, k_tmp);
      a01 = m_entry(A, k_tmp, k_max);
      a10 = m_entry(A, k_max, k_tmp);
      a11 = m_entry(A, k_max, k_max);

      /* treat degenerate cases differently
         -- if there are still no splits after five iterations
            and the bottom 2 x 2 looks degenerate, force it to
         split */
      #ifdef DEBUG
        printf("# schur: bottom 2 x 2 = [%lg, %lg; %lg, %lg]\n",
               a00, a01, a10, a11);
      #endif
      if(iter >= 5 &&
         fabs(a00-a11) < sqrt_macheps*(fabs(a00)+fabs(a11)) &&
         (fabs(a01) < sqrt_macheps*(fabs(a00)+fabs(a11)) ||
          fabs(a10) < sqrt_macheps*(fabs(a00)+fabs(a11))) )
      {
        if(fabs(a01) < sqrt_macheps*(fabs(a00)+fabs(a11)))
          m_set_val(A, k_tmp, k_max, 0.0);
        if(fabs(a10) < sqrt_macheps*(fabs(a00)+fabs(a11)))
        {
          m_set_val(A, k_max, k_tmp, 0.0);
          split = 1;
          continue;
        }
      }

      s = a00 + a11;
      t = a00*a11 - a01*a10;

      /* break loop if a 2 x 2 complex block */
      if(k_max == k_min + 1 && s*s < 4.0*t)
      {
        split = 1;
        continue;
      }

      /* perturb shift if convergence is slow */
      if((iter % 10) == 0)
      {
        s += iter*0.02;
        t += iter*0.02;
      }

      /* set up Householder transformations */
      k_tmp = k_min + 1;

      a00 = m_entry(A, k_min, k_min);
      a01 = m_entry(A, k_min, k_tmp);
      a10 = m_entry(A, k_tmp, k_min);
      a11 = m_entry(A, k_tmp, k_tmp);

      x = a00*a00 + a01*a10 - s*a00 + t;
      y = a10*(a00+a11-s);
      if(k_min + 2 <= k_max)
        z = a10*A->me[k_min+2][k_tmp];
      else
        z = 0.0;

      for(k = k_min; k <= k_max-1; k++)
      {
        if(k < k_max - 1)
        {
          hhldr3(x, y, z, &nu1, &beta2, &dummy);
          if(Q != MNULL)
            hhldr3rows(Q, k, n-1, beta2, nu1, y, z);
        }
        else
        {
          givens(x, y, &c, &s);
          rot_cols(A, k, k+1, c, s, A);
          rot_rows(A, k, k+1, c, s, A);
          if(Q)
            rot_cols(Q, k, k+1, c, s, Q);
        }
        x = m_entry(A, k+1, k);
        if(k <= k_max - 2)
          y = m_entry(A, k+2, k);
        else
          y = 0.0;
        if(k <= k_max - 3)
          z = m_entry(A, k+3, k);
        else
          z = 0.0;
      }
	  for(k = k_min; k <= k_max-2; k++)
	  {
        /* zero appropriate sub-diagonals */
        m_set_val(A, k+2, k, 0.0);
        if(k < k_max-2)
	      m_set_val(A, k+3, k, 0.0);
      }

      /* test to see if matrix should split */
      for(k = k_min; k < k_max; k++)
        if(fabs(A_me[k+1][k]) < MACHEPS*
          (fabs(A_me[k][k])+fabs(A_me[k+1][k+1])))
        {
          A_me[k+1][k] = 0.0;
          split = 1;
        }
	}
  }

  /* polish up A by zeroing strictly lower triangular elements
     and small sub-diagonal elements */
  for(i = 0; i < A->m; i++)
    for(j = 0; j < i-1; j++)
      A_me[i][j] = 0.0;
    for(i = 0; i < A->m - 1; i++)
      if(fabs(A_me[i+1][i]) < MACHEPS*
         (fabs(A_me[i][i])+fabs(A_me[i+1][i+1])))
        A_me[i+1][i] = 0.0;

#ifdef THREADSAFE
  V_FREE(diag);  V_FREE(beta);
#endif

  return A;
}

/* schur_vals -- compute real & imaginary parts of eigenvalues
	-- assumes T contains a block upper triangular matrix
		as produced by schur()
	-- real parts stored in real_pt, imaginary parts in imag_pt */
#ifndef ANSI_C
void schur_evals(T, real_pt, imag_pt)
MAT	*T;
VEC	*real_pt, *imag_pt;
#else
void schur_evals(MAT *T, VEC *real_pt, VEC *imag_pt)
#endif
{
  int i, n;
  double discrim, **T_me;
  double diff, sum, tmp;

  n = T->n;	T_me = T->me;
  real_pt = v_resize(real_pt, (unsigned int)n);
  imag_pt = v_resize(imag_pt, (unsigned int)n);

  i = 0;
  while(i < n)
  {
    if(i < n-1 && T_me[i+1][i] != 0.0)
    {   /* should be a complex eigenvalue */
      sum  = 0.5*(T_me[i][i]+T_me[i+1][i+1]);
      diff = 0.5*(T_me[i][i]-T_me[i+1][i+1]);
      discrim = diff*diff + T_me[i][i+1]*T_me[i+1][i];
      if(discrim < 0.0)
      { /* yes -- complex e-vals */
        real_pt->ve[i] = real_pt->ve[i+1] = sum;
        imag_pt->ve[i] = sqrt(-discrim);
        imag_pt->ve[i+1] = - imag_pt->ve[i];
      }
      else
      { /* no -- actually both real */
        tmp = sqrt(discrim);
        real_pt->ve[i]   = sum + tmp;
        real_pt->ve[i+1] = sum - tmp;
        imag_pt->ve[i]   = imag_pt->ve[i+1] = 0.0;
      }
      i += 2;
    }
    else
    {   /* real eigenvalue */
      real_pt->ve[i] = T_me[i][i];
      imag_pt->ve[i] = 0.0;
      i++;
    }
  }
}

/* m_get_eigenvalues -- get the eigenvalues of a matrix A
	-- */
CMPLX *m_get_eigenvalues(MAT *A)
{
  MAT *T = MNULL, *Q = MNULL;
  VEC *evals_re = VNULL, *evals_im = VNULL;

  CMPLX *z;

  Q = m_get(A->m, A->n);
  T = m_copy(A, MNULL);
  /* compute Schur form: A = Q.T.Q^T */
  schur(T, Q);
  /* extract eigenvalues */
  evals_re = v_get(A->m);
  evals_im = v_get(A->m);
  schur_evals(T, evals_re, evals_im);

  z = malloc(evals_re->dim*sizeof(CMPLX));
  for(int i = 0; i < evals_re->dim; i++)
  {
//    z[i] = evals_re->ve[i] + I*evals_im->ve[i];
    z[i].real = evals_re->ve[i];
    z[i].imag = evals_im->ve[i];
  }
  return z;
}

double cmplx_mag(double real, double imag)
{
  return sqrt(real * real + imag * imag);
}

/* max_mag_eigenvalue -- extracts the magnitude of the maximum eigenvalue
	-- */
double max_mag_eigenvalue(CMPLX *z, int size)
{
  double maximum = 0, aux;
  for(int c = 1; c < size; c++)
  {
    aux = cmplx_mag(z[c].real, z[c].imag);
    if(aux > maximum)
    {
      maximum  = aux;
    }
  }
  return (double)maximum;
}

/* is_same_sign -- check if a has the same sign as b
	-- */
int is_same_sign(double a, double b)
{
  if(((a >= 0) && (b >= 0)) || ((a <= 0) && (b <= 0)))
    return 1;
  else
    return 0;
}

/* y_k -- computes the output signal in the k-th sample
	-- */
double y_k(MAT *A, MAT *B, MAT *C, MAT *D, double u, int k, MAT *x0)
{
  MAT *y = MNULL/* *U*/, *Ak = MNULL, *AUX = MNULL, *AUX2 = MNULL;
  MAT *AUX3 = MNULL, *AUX4 = MNULL, *AUX5 = MNULL;
//  U = m_get(A->m, A->n);
//  U = m_same_elements(U,u);
  // y = C * A.pow(k) * x0;
  Ak = m_get(A->m, A->n);
  Ak = m_pow(A, k, MNULL);
  AUX = m_get(A->m, A->n);
  AUX = m_mlt(C, Ak, MNULL);
  y = m_get(A->m, A->n);
  y = m_mlt(AUX, x0, MNULL);

  AUX2 = m_get(A->m, A->n);
  for(int m = 0; m <= (k - 1); m++)
  {
    // y += (C * A.pow(k - m - 1) * B * u) + D * u;
    Ak = m_pow(A, (k-m-1), MNULL);
    AUX = m_mlt(C, Ak, MNULL);
    AUX2 = m_mlt(AUX, B, MNULL);
//    AUX3 = m_mlt(AUX2, U, MNULL);
//    AUX4 = m_mlt(D, U, MNULL);
    AUX5 = m_add(AUX2, D, MNULL);
    y = m_add(y, AUX5, MNULL);
  }
  return y->me[0][0]*u;
}

/* peak_output -- computes the biggest peak value of a signal (Mp)
	-- */
void peak_output(MAT *A, MAT *B, MAT *C, MAT *D, MAT *x0,
                 double *out, double yss, double u)
{
  double greater;
  int i = 0;
  greater = fabs(y_k(A, B, C, D, u, i, x0));
  while((fabs(y_k(A, B, C, D, u, i+1, x0)) >= fabs(yss)))
  {
    if(greater < fabs(y_k(A, B, C, D, u, i+1, x0)))
    {
      greater = fabs(y_k(A, B, C, D, u, i+1, x0));
      out[1] = y_k(A, B, C, D, u, i+1, x0);
      out[0] = i+2;
    }
    if(!is_same_sign(yss, out[1]))
    {
      greater = 0;
    }
    i++;
  }
}

double y_ss(MAT *A, MAT *B, MAT *C, MAT *D, double u)
{
  double yss;
  MAT *AUX, *AUX2, *AUX3, *AUX4, *AUX5;
  MAT *Id;

  // get the expression y_ss=(C(I-A)^(-1)B+D)u
  Id = m_get(A->m, A->n);
  Id = m_ident(Id);
  AUX = m_get(A->m, A->n);
  // Id - A
  AUX = m_sub(Id, A, MNULL);
  AUX2 = m_get(A->m, A->n);
  AUX2 = m_inverse(AUX, MNULL);
  AUX3 = m_get(A->m, A->n);
  AUX3 = m_mlt(C, AUX2, MNULL);
  AUX4 = m_get(A->m, A->n);
  AUX4 = m_mlt(AUX3, B, MNULL);
  AUX5 = m_get(A->m, A->n);
  AUX5 = m_add(AUX4, D, MNULL);
  yss = AUX5->me[0][0] * u;

  return yss;
}

double c_bar(double mp, double yss, double lambmax, int kp)
{
  double cbar;
  cbar = (mp-yss)/(pow(lambmax, kp));
  return cbar;
}

double log_b(double base, double x)
{
  return (double) (log(x) / log(base));
}

int k_bar(double lambdaMax, double p, double cbar, double yss, int order)
{
  double k_ss, x;
  x = (p * yss) / (100 * cbar);
  k_ss = log_b(lambdaMax, x);
  return ceil(k_ss)+order;
}

double max_mag_eigenvalue2(MAT *A)
{
  double maximum = 0, aux;
  CMPLX *z;
  z = m_get_eigenvalues(A);
  for(int i = 0; i < A->m; i++)
  {
    aux = cmplx_mag(z[i].real, z[i].imag);
    if(aux > maximum)
    {
      maximum = aux;
    }
  }
  return maximum;
}

int check_settling_time(MAT *A, MAT *B, MAT *C, MAT *D, MAT *x0,
                        double u, double tsr, double p, double ts)
{
  double peakV[2];
  double yss, mp, lambMax, cbar, output;
  int kbar, kp, i;
  yss = y_ss(A, B, C, D, u);
  peak_output(A, B, C, D, x0, peakV, yss, u);
  mp = (double) peakV[1];
  kp = (int) peakV[0];
  lambMax = max_mag_eigenvalue2(A);
  printf("Mp=%f", mp);
  printf("yss=%f", yss);
  printf("lambMax=%f", lambMax);
  printf("kp=%d", kp);

  cbar = c_bar(mp, yss, lambMax, kp);

  kbar = k_bar(lambMax, p, cbar, yss, A->m);
  printf("cbar=%f", cbar);
  if(kbar * ts < tsr)
  {
    //printf("kbar=%f", kbar);
    return 1;
  }

  i = ceil(tsr / ts);
  while(i <= kbar)
  {
    output = y_k(A, B, C, D, u, i, x0);
    if(!(output > (yss - (yss * (p/100))) && (output < (yss * (p/100) + yss))))
    {
      //printf("kbar=%f", kbar);
      return 0;
    }
    i++;
  }
  //printf("kbar=%f", kbar);
  return 1;
}

int main(){
    MAT *A = MNULL, *A2 = MNULL, *A3 = MNULL, *A4 = MNULL, *A5 = MNULL, *A6 = MNULL, *B = MNULL, *C = MNULL, *D = MNULL, *T = MNULL, *Q = MNULL, *X_re = MNULL, *X_im = MNULL, *Q1 = MNULL, *Q1_inv = MNULL;
    MAT *Q1_temp, *Test = MNULL;
//    VEC *evals_re = VNULL, *evals_im = VNULL;
    MAT *F = MNULL, *G = MNULL, *H = MNULL;
    int k=3;
    double y, x0;
    CMPLX *z;
    //ZMAT *ZQ = ZMNULL, *ZQ_temp, *ZQ_inv = ZMNULL, *ZH, *ZF;

   //setting up A matrix
//    A=m_get(4,4);
//    A->me[0][0]=-0.5000;A->me[0][1]=0.6000;A->me[0][2]=0;A->me[0][3]=0;
//    A->me[1][0]=-0.6000;A->me[1][1]=-0.5000;A->me[1][2]=0;A->me[1][3]=0;
//    A->me[2][0]=0;A->me[2][1]=0;A->me[2][2]=0.2000;A->me[2][3]=0.8000;
//    A->me[3][0]=0;A->me[3][1]=0;A->me[3][2]=-0.8000;A->me[3][3]=0.2000;printf("A ");m_output(A);
    A=m_get(5,5);
    A->me[0][0]=-0.5000;A->me[0][1]=0.6000;A->me[0][2]=0;A->me[0][3]=0;A->me[0][4]=0;
    A->me[1][0]=-0.6000;A->me[1][1]=-0.5000;A->me[1][2]=0;A->me[1][3]=0;A->me[1][4]=0;
    A->me[2][0]=0;A->me[2][1]=0;A->me[2][2]=0.2000;A->me[2][3]=0.8000;A->me[2][4]=0;
    A->me[3][0]=0;A->me[3][1]=0;A->me[3][2]=-0.8000;A->me[3][3]=0.2000;A->me[3][4]=0;
    A->me[4][0]=0;A->me[4][1]=0;A->me[4][2]=0;A->me[4][3]=0;A->me[4][4]=0.6;printf("A ");
    m_output(A);
    A2=m_get(5,5);
    A2 = m_add(A, A, A2);printf("A+A=\n");
    m_output(A2);
    A3=m_get(5,5);
    A3 = m_sub(A, A, A3);printf("A-A=\n");
    m_output(A3);
    A4=m_get(5,5);
    A4 = m_mlt(A, A, A4);printf("A*A=\n");
    m_output(A4);
	A5=m_get(5,5);
    A5 = m_inverse(A,A5);printf("inv(A)=\n");
    m_output(A5);
	A6=m_get(5,5);
    A6 = m_pow(A,50,A6);printf("pow(A)=\n");
    m_output(A6);

	z = m_get_eigenvalues(A);
	int size = A->m;
	printf("size=%d\n",size);
	for(int i=0;i<size;i++){
//		printf("%f+%f i", z[i].real, z[i].imag);
        printfc(z[i]);
	}

	printf("Maximum:%f\n", max_mag_eigenvalue(z, size));

//    printf("testing /n");
    //setting up B matrix
//    B=m_get(4,1);
//    B->me[0][0]=0;
//    B->me[1][0]=0;
//    B->me[2][0]=2.5;
//    B->me[3][0]=1;printf("B ");m_output(B);
    /*B=m_get(5,1);
    B->me[0][0]=0;
    B->me[1][0]=0;
    B->me[2][0]=2.5;
    B->me[3][0]=1;
    B->me[4][0]=0;printf("B ");m_output(B);*/
    //setting up C matrix
//    C=m_get(1,4);
//    C->me[0][0]=0;C->me[0][1]=2.6;C->me[0][2]=0.5;C->me[0][3]=1.2;printf("C ");m_output(C);
        /*C=m_get(1,5);
        C->me[0][0]=0;C->me[0][1]=2.6;C->me[0][2]=0.5;C->me[0][3]=1.2;C->me[0][4]=0;printf("C ");m_output(C);*/
    //setting up D matrix
    /*D=m_get(1,1);
    D->me[0][0]=0;printf("D ");m_output(D);
    printf("-----------------------------------------------------------\n");
    printf("k_ss=%d\n",k_ss(A,B,C,D,5,1.0f));
    Test = m_pow(A,2,Test);
	m_output(Test);*/

//    /* read in A matrix */
//    printf("Input A matrix:\n");
//
//    A = m_input(MNULL);     /* A has whatever size is input */
//    //B = m_input(MNULL);     /* B has whatever size is input */
//
//    if ( A->m < A->n )
//    {
//        printf("Need m >= n to obtain least squares fit\n");
//        exit(0);
//    }
//    printf("# A =\n");       m_output(A);
//
//    //zm_output(zm_A_bar(A));
//
//   Q = m_get(A->m,A->n);
//   T = m_copy(A,MNULL);
//   printf("A=:%f\n",A->me[0][0]);
//   printf("T=:%f\n",T->me[0][0]);
//   /* compute Schur form: A = Q.T.Q^T */
//   schur(T,Q);
//   /* extract eigenvalues */
//   evals_re = v_get(A->m);
//   evals_im = v_get(A->m);
//   printf("A=:%f\n",A->me[0][0]);
//   printf("T=:%f\n",T->me[0][0]);
//   schur_evals(T,evals_re,evals_im);
//   printf("A=:%f\n",A->me[0][0]);
//   printf("T=:%f\n",T->me[0][0]);
//   printf("test=:%f\n",evals_re->ve[0]);
//   printf("test=:%f\n",evals_re->ve[1]);
//   printf("test=:%f\n",evals_re->ve[2]);
//   printf("test=:%f\n",evals_re->ve[3]);
//   printf("test=:%f\n",evals_re->ve[4]);
//
//   z=malloc(evals_re->dim*sizeof(complex double));
//   for(int i=0;i<evals_re->dim;i++){
//   	z[i]=evals_re->ve[i]+I*evals_im->ve[i];
//   	printf("Z[%d]=%f + i%f\n", i, creal(z[i]), cimag(z[i]));
//   }
//
//   size_t size=(size_t)sizeof(z);
//   printf("Maximum:%f\n",max_mag_eigenvalue(z,size));
  return 0;
}
