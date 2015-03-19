#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define N 1
int main() {
  double *dev_a = (double*)malloc(sizeof(double));
  double a = 1;
  memcpy(dev_a,&a,sizeof(double));
  assert(dev_a[0]==1);
  return 0;
}

