// #7906: a statement expression's last value is used, and a vector does not
// decay to a pointer (C11 6.3.2.1p3 covers arrays only).
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  v4i x = ({
    v4i t = {1, 2, 3, 4};
    t;
  });
  assert(x[0] == 1 && x[3] == 4);

  // The value survives arithmetic inside the statement expression.
  v4i y = ({
    v4i a = {10, 20, 30, 40};
    v4i b = {1, 2, 3, 4};
    a + b;
  });
  assert(y[0] == 11 && y[3] == 44);

  // An array still decays, which is what the rewrite exists for.
  int arr[4] = {5, 6, 7, 8};
  int *p = ({
    arr;
  });
  assert(p[2] == 7);
  return 0;
}
