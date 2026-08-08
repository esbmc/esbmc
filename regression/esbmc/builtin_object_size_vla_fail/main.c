#include <assert.h>

unsigned int nondet_uint();

/* The companion to builtin_object_size_vla: a fixed-size array still gets a
   real answer, so the unknown-size fallback has not swallowed the ordinary
   case. Asserting the wrong size must fail. */
int main()
{
  unsigned int n = nondet_uint();
  __ESBMC_assume(n >= 1 && n <= 8);

  char fixed[16];
  char vla[n];
  vla[0] = 'x';

  assert(__builtin_object_size(fixed, 0) == 16);
  /* Wrong on purpose: the object is 16 bytes, not 8. */
  assert(__builtin_object_size(fixed, 0) == 8);
  return 0;
}
