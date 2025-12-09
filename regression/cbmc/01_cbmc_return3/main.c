#include <stddef.h>

// If inlined, irep2-migrating ESBMC would forget to evaluate the return
// value if the return value was discarded. This eliminated an illegal
// behavior. It's a legitimate optimization, but not guaranteed by the
// spec.
inline int
foo(int *bar)
{
  return *bar;
}

int
main()
{
  foo(NULL);
  return 0;
}
