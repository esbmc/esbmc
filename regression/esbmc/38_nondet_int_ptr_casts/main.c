#include <stdbool.h>

int
main()
{
  int bees, fleas;
  unsigned int foo = (unsigned int*)&bees;
  unsigned int bar = (unsigned int*)&fleas;

  bool baz = nondet_bool();
  __ESBMC_assume(baz == true);
  unsigned int qux = (baz) ? foo : bar;
  int *fin = (int *)qux;
  assert (fin == &bees);
  *fin = 0;

  assert(bees == 0);
  return 0;
}
