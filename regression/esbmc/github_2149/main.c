#include <assert.h>
#include <stdatomic.h>

atomic_uint y = 3;
atomic_int w = -2;

int main()
{
  unsigned int z = 2 * y; /* CK_AtomicToNonAtomic: read atomic_uint as uint */
  y = z + 1;              /* CK_NonAtomicToAtomic: write uint back to atomic_uint */
  assert(z == 6);

  int v = w + 1;          /* CK_AtomicToNonAtomic: read atomic_int as int */
  w = v - 1;              /* CK_NonAtomicToAtomic: write int back to atomic_int */
  assert(v == -1);

  return 0;
}
