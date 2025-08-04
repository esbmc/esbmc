//#include <stdio.h>
#include <assert.h>

#define XLen 10

unsigned int nondet_uint();
//unsigned int XLen=nondet_uint();

float y(int n, float *x) {
  if (n<0) return 0;
  else {
//    return (0.4375*y(n-1)+0.125*y(n-2)+x[n]);
    return (0.5*y(n-1,x)+x[n]);
  }
}
/*
 this is added to help the invariant
*/

float sum_y(int k, float *x) {
  float sum = 0;
  for(int j = 0; j < k; j++) {
    sum += y(j, x);
  }
  return sum;
}

int main() {

  int i;
  float total=0;
//  float x[] = {1,0,0,0,0,0,0,0,0,0};
//  total = y(0,x)+y(1,x)+y(2,x)+y(3,x)+y(4,x)+y(5,x)+y(6,x)+y(7,x)+y(8,x)+y(9,x);   
#if 1
  float x[XLen];
  x[0]=1;
  for(i=1; i<XLen; i++)
	  x[i]=0;

  __ESBMC_loop_invariant(i == 0 ? total == 0 : total == sum_y(i,x)); // only the invariant is not enough, need a function to help the invariant
  for(i=0; i<XLen; i++) {
    printf("%f\n", y(i,x));
    total+=y(i,x);
  }
#endif
  assert(total<=2);
  printf("total: %f\n", total);
  return 0;
}

