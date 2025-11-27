// Test --unwindsetname with C++ namespaces
// Different namespaces with same function name should be distinguishable

#include <cassert>

namespace ModuleA {
  void process() {
    int i, limit;
    int sum = 0;

    __ESBMC_assume(limit == 10);

    for (i = 0; i < limit; i++) {
      sum += i;
    }

    assert(sum == 45);
  }
}

namespace ModuleB {
  void process() {
    int j, limit;
    int prod = 1;

    __ESBMC_assume(limit == 5);

    for (j = 0; j < limit; j++) {
      prod *= 2;
    }

    assert(prod == 32);
  }
}

int main() {
  ModuleA::process();
  ModuleB::process();
  return 0;
}
