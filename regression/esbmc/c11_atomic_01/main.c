// C11 _Atomic type: init, implicit casts between atomic and non-atomic
#include <assert.h>

int main()
{
  _Atomic int x = 5;
  int val = x;            // AtomicToNonAtomic
  x = val + 1;            // NonAtomicToAtomic
  int result = x;
  assert(result == 6);

  _Atomic long al = 100L;
  long l = al;
  assert(l == 100L);

  return 0;
}
