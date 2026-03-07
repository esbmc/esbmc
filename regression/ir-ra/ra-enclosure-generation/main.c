#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main() {
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y;


  assert(z > x);
}
