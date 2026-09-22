/* Two claims violated at different k. Both belong in one table, and the
   forward condition's unwinding assertion is not a property of the program. */
#include <assert.h>

int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  assert(n != 7);
  for (unsigned i = 0; i < n; ++i)
    ++x;
  assert(x != 3);
}
