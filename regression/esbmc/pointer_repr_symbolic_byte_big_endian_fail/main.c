#include <assert.h>

struct S
{
  int *p;
  char buf[8];
};

unsigned nondet_uint();
int *nondet_ptr();

// A byte written at a symbolic index may land in p, after which p is no longer
// the pointer stored there.
int main()
{
  int *q = nondet_ptr();
  struct S s;
  s.p = q;
  unsigned char *bytes = (unsigned char *)&s;
  bytes[nondet_uint() % sizeof(s)] = 0xff;
  assert(s.p == q);
}
