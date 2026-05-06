
#ifdef __ESBMC_FIXEDBV

#  include <math.h>

#  undef sqrt

/*Returns the square root of n. Note that the function */
/*Babylonian method*/
/*http://www.geeksforgeeks.org/square-root-of-a-perfect-square/*/
double sqrt(double n)
{
__ESBMC_HIDE:;
  /*We are using n itself as initial approximation
   This can definitely be improved */
  double x = n;
  double y = 1;
  double e = 1;
  int i = 0;
  while (i++ < 15) //Change this line to increase precision
  {
    x = (x + y) / 2.0;
    y = n / x;
  }
  return x;
}

#endif
