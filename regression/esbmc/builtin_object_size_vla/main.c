#include <assert.h>

unsigned int nondet_uint();

/* __builtin_object_size on a VLA. The array has no compile-time size, so
   type_byte_size() cannot produce one and throws; GCC answers "unknown" for
   exactly these objects. Letting the exception escape aborted the run with
   array_type2t::dyn_sized_array_excp before symex reached any property. */
int main()
{
  unsigned int n = nondet_uint();
  __ESBMC_assume(n >= 1 && n <= 8);

  char vla[n];
  vla[0] = 'x';

  /* Type 0 is the "maximum size" query, so an unknown answer is a large
     value rather than zero. */
  __ESBMC_assume(__builtin_object_size(vla, 0) != 0);
  /* Type 2 is the "minimum size" query, whose unknown answer is zero. */
  assert(__builtin_object_size(vla, 2) == 0);

  assert(vla[0] == 'x');
  return 0;
}
