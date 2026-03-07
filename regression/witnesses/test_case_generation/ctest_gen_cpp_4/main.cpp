#include <cassert>

extern "C" int __VERIFIER_nondet_int(void);

int main() {
  int x = __VERIFIER_nondet_int();
  assert(x != 42);
  return 0;
}
