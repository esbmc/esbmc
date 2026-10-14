// github #6494, known limitation: a program may replace operator new and leave
// operator delete alone. [new.delete.single]/10 lets the default deallocation
// take storage "allocated by an earlier call to a (possibly replaced) operator
// new", so this program is well defined. Storage from the replacement is not
// one of ESBMC's built-in dynamic objects though, so the delete -- which stays
// on the built-in path, there being no replacement to call -- reports
// "Mismatched memory deallocation operators".
//
// Routing the allocation only when a replaced deallocation exists would fix
// this, but it also disables the pool-aliasing detection this issue is about
// for programs that never delete (gcc-template-tests/new11), so the alarm
// stands until the built-in deallocation can accept foreign storage.
#include <cstddef>
#include <cassert>

static char pool[64];

void *operator new(size_t)
{
  return pool;
}

int main()
{
  int *p = new int(1);
  assert(*p == 1);
  delete p;
  return 0;
}
