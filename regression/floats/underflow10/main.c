#include <assert.h>
#include <float.h>

int main() {
  double a = DBL_TRUE_MIN * 2;
  double b = DBL_TRUE_MIN;
  double result = a - b;

  assert(result > 0.0);
  return 0;
}

