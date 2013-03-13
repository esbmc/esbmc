#include <math.h>
#include <stdio.h>
#include <assert.h>


int main() {
  printf("%.16f \n", cos(45));
  printf("%.16f \n", fabs(cos(45) - 0.5253219888177297));

  assert(fabs(cos(45) - 0.5253219888177297) <= 1e-8 );
  return 0;
}

