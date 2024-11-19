#include <stdatomic.h>
#include <assert.h>

int main()
{
  // NonAtomicToAtomic
  _Atomic(int) atomic_var = 42;
  // AtomicToNonAtomic
  int non_atomic_var = atomic_var;
  assert(non_atomic_var == 42);

  return 0;
}
