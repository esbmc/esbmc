#include <math.h>
#include <stdio.h>
#include <assert.h>

#define M_PI     3.14159265358979323846
int main() {
  printf("%.16f \n", cos(45));
  assert(fabs(cos(45) - 0.5253219888177297) <= 1e-8 );
#if 0
  printf("%.16f \n", cos(M_PI/4));
  printf("%.16f \n", fabs(cos(M_PI/4) - 0.7071067811865476));
  printf("%.16f \n", cos(1600*M_PI/4));
  printf("%.16f \n", fabs(cos(1600*M_PI/4) - 1.0000000000000000));
  printf("%.16f \n", cos(16000.5466546546546*M_PI/4));
  printf("%.16f \n", fabs(cos(16000.5466546546546*M_PI/4) - 0.9092400362880396));
  printf("%.16f \n", cos(16000.5466546546546/4));
  printf("%.16f \n", fabs(cos(16000.5466546546546/4) + 0.6300213338984322));


  assert(fabs(cos(M_PI/4) - 0.7071067811865476) <= 1e-8 );
  assert(fabs(cos(1600*M_PI/4) - 1.0000000000000000) <= 1e-8 );
  assert(fabs(cos(16000.5466546546546*M_PI/4) - 0.9092400362880396) <= 1e-8 );
  assert(fabs(cos(16000.5466546546546/4) + 0.6300213338984322) <= 1e-8 );
#endif
  return 0;
}

