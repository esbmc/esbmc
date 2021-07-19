#include <math.h>

int a = 42;

int main() {
  if(!__builtin_constant_p(a)) __VERIFIER_error();
  return 0;
}