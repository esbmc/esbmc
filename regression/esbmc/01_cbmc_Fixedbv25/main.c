#include <assert.h>

int roundInt(float a) {
  if (a>=0)
    return (int)(a+0.5f);
  else
    return (int)(a-0.5f);
}

void main() {
  float a[1];
	a[0] = -1.0f;
  assert(roundInt(a[0]) == -1.0);
}

