#include <assert.h>

int roundInt(float a) {
  if (a>=0)
    return (int)(a+0.5f);
  else
    return (int)(a-0.5f);
}

void main() {
  float a[1];
	int temp;
	//a[0] = -0.6;
	a[0] = -12.5;
//  printf("%d\n", roundInt(a));
  assert(roundInt(a[0]) == -12.0);
}

