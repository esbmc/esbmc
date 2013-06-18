#include <math.h>
#include <stdio.h>
#include <assert.h>

int main() {
  printf("%.16f \n", exp(0.3));
  printf("%.16f \n", exp(0.14115125));
  printf("%.16f \n", exp(-2.132314121515));
  printf("%.16f \n", exp(3.1123441));
  printf("%.16f \n", exp(-10.34));

  assert(fabs(exp(0.3) - 1.3498588075760032) <= 1e-8 );
  assert(fabs(exp(0.14115125) - 1.1515988141337348) <= 1e-9 );
  assert(fabs(exp(-2.132314121515) - 0.11856260786488046) <= 1e-1 );
  assert(fabs(exp(-10.34) - 3.231432266044366e-05) <= 1e-4);
  return 0;
}

