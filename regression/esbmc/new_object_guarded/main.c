#include <assert.h>

struct S
{
  int a;
  int *q;
};

struct S *__ESBMC_new_object(void);
int nondet_int(void);

int main()
{
  int x = 5;
  int n = nondet_int();
  if (n > 0)
  {
    // Allocated under a path guard: the binding of s must still be the
    // object's address, so the field writes and reads below resolve to it.
    struct S *s = __ESBMC_new_object();
    s->q = &x;
    s->a = n;
    assert(*s->q == 5);
    assert(s->a > 0);
  }
  return 0;
}
