#include <stdlib.h>

int
main()
{
  void *beans = malloc(nondet_uint());
  assert(beans != NULL); // Null is possible if nondet is zero; fail.
  return 0;
}
