extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "ludcmp.c", 3, "reach_error"); }

void __VERIFIER_assert(int cond) {
  if (!(cond)) {
    ERROR: {reach_error();abort();}
  }
  return;
}
/*************************************************************************/
/*                                                                       */
/*   SNU-RT Benchmark Suite for Worst Case Timing Analysis               */
/*   =====================================================               */
/*                              Collected and Modified by S.-S. Lim      */
/*                                           sslim@archi.snu.ac.kr       */
/*                                         Real-Time Research Group      */
/*                                        Seoul National University      */
/*                                                                       */
/*                                                                       */
/*        < Features > - restrictions for our experimental environment   */
/*                                                                       */
/*          1. Completely structured.                                    */
/*               - There are no unconditional jumps.                     */
/*               - There are no exit from loop bodies.                   */
/*                 (There are no 'break' or 'return' in loop bodies)     */
/*          2. No 'switch' statements.                                   */
/*          3. No 'do..while' statements.                                */
/*          4. Expressions are restricted.                               */
/*               - There are no multiple expressions joined by 'or',     */
/*                'and' operations.                                      */
/*          5. No library calls.                                         */
/*               - All the functions needed are implemented in the       */
/*                 source file.                                          */
/*                                                                       */
/*                                                                       */
/*************************************************************************/
/*                                                                       */
/*  FILE: ludcmp.c                                                       */
/*  SOURCE : Turbo C Programming for Engineering                         */
/*                                                                       */
/*  DESCRIPTION :                                                        */
/*                                                                       */
/*     Simultaneous linear equations by LU decomposition.                */
/*     The arrays a[][] and b[] are input and the array x[] is output    */
/*     row vector.                                                       */
/*     The variable n is the number of equations.                        */
/*     The input arrays are initialized in function main.                */
/*                                                                       */
/*                                                                       */
/*  REMARK :                                                             */
/*                                                                       */
/*  EXECUTION TIME :                                                     */
/*                                                                       */
/*                                                                       */
/*************************************************************************/





/*
** Benchmark Suite for Real-Time Applications, by Sung-Soo Lim
**
**    III-4. ludcmp.c : Simultaneous Linear Equations by LU Decomposition
**                 (from the book C Programming for EEs by Hyun Soon Ahn)
*/



double a[50][50], b[50], x[50];

int ludcmp(int nmax, int n, double eps);


static double fabs(double n)
{
  double f;

  if (n >= 0) f = n;
  else f = -n;
  return f;
}

int main()
{

	int i, j, nmax = 50, n = 5, chkerr;
	double eps, w;

	eps = 1.0e-6;

	for(i = 0; i <= n; i++)
	{
			w = 0.0;
			for(j = 0; j <= n; j++)
			{
					a[i][j] = (i + 1) + (j + 1);
					if(i == j) a[i][j] *= 10.0;
					w += a[i][j];
			}
                        __VERIFIER_assert(i==2);
			b[i] = w;
	}

	chkerr = ludcmp(nmax, n, eps);

    return 0;
}

int ludcmp(int nmax, int n, double eps)
{

	int i, j, k;
	double w, y[100];

	if(n > 99 || eps <= 0.0) return(999);
	for(i = 0; i < n; i++)
	{
			if(fabs(a[i][i]) <= eps) return(1);
			for(j = i+1; j <= n; j++)
			{
			  w = a[j][i];
			  if(i != 0)
			    for(k = 0; k < i; k++)
			      w -= a[j][k] * a[k][i];
			  a[j][i] = w / a[i][i];
			}
			for(j = i+1; j <= n; j++)
			  {
			    w = a[i+1][j];
			    for(k = 0; k <= i; k++)
			      w -= a[i+1][k] * a[k][j];
			    a[i+1][j] = w;
			  }
	}
	y[0] = b[0];
	for(i = 1; i <= n; i++)
	  {
	    w = b[i];
	    for(j = 0; j < i; j++)
	      w -= a[i][j] * y[j];
	    y[i] = w;
	  }
	x[n] = y[n] / a[n][n];
	for(i = n-1; i >= 0; i--)
	  {
	    w = y[i];
	    for(j = i+1; j <= n; j++)
	      w -= a[i][j] * x[j];
	    x[i] = w / a[i][i] ;
	  }
	return(0);

}