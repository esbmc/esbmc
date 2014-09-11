#include <math.h>
#include <stdio.h>
#include <assert.h>


int main() {
  printf("%.16f \n", exp(-45));
  printf("%.16f \n", exp(-0.14115125));
  printf("%.16f \n", exp(2.132314121515));
  printf("%.16f \n", exp(-3.1123441));
  printf("%.16f \n", exp(-100.34));

  assert(fabs(exp(-45) - 0.0000000000000000) <= 1e-8 );
  assert(fabs(exp(-0.14115125) - 0.8683579626227979) <= 1e-9 );
  assert(fabs(exp(2.132314121515) - 8.4343623846368754) <= 1e-8 );
  assert(fabs(exp(-3.1123441) - 0.0444965286819433) <= 1e-7);
  assert(fabs(exp(-100.34) - 0.0000000000000000) <= 1e-9);
  return 0;
}

