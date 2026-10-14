#include <assert.h>

/* The counterexample decodes a union's short member from its lowest address,
   as the verdict does. */
union U
{
  unsigned i;
  unsigned short s;
};

union U nondet_U();

int main()
{
  union U h = nondet_U();
  __ESBMC_assume(h.i == 0x01020304u);
  assert(h.s != 0x0102);
  return 0;
}
