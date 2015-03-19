#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define N 1
int main() {
  int *dev_a = (int*)malloc(sizeof(int));
  int a = 2;
  memcpy(dev_a,&a,sizeof(int));
  assert(dev_a[0]==1);
  return 0;
}

