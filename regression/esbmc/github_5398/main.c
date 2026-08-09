// Whichever alternative C17 7.22.3p1 hands back, free() disposes of it: the
// NULL arm is a no-op and the success arm releases the object. Before the fix
// the option returned NULL without assigning it, leaving p unconstrained, and
// this reported "invalid pointer freed".
#include <stdlib.h>

int main(void)
{
  void *p = malloc(0);
  free(p);
  return 0;
}
