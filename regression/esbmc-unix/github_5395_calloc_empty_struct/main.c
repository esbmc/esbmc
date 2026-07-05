// Regression for github #5395: a zero-sized calloc request must honour
// --force-malloc-success and return non-null, exactly like malloc(0).
//
// Here `struct bus_info` is empty, so sizeof == 0 and kzalloc()/calloc()
// request zero bytes.  Previously calloc(_, 0) returned NULL
// unconditionally (ignoring --force-malloc-success), so the NULL-check
// error path was wrongly reachable and reach_error fired -- a false
// alarm on the unreach-call property.  The original benchmark is
// c/ldv-regression/rule57_ebda_blast.i.
#include <stdlib.h>

struct bus_info {};

void reach_error(void) {}

static void *kzalloc(int size, int flags)
{
  (void)flags;
  return calloc(1, size); // calloc(1, 0) when size == 0
}

int main(void)
{
  struct bus_info *p = (struct bus_info *)kzalloc(sizeof(struct bus_info), 0);
  if (!p)
    reach_error(); // unreachable under --force-malloc-success
  return 0;
}
