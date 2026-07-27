#include <assert.h>

int main() {
  _Complex double a = 1.0 + 2.0i;
  _Complex double b = 3.0 - 1.0i;
  _Complex double m = a * b;
  assert(__imag__ m == 4.0);
  return 0;
}
