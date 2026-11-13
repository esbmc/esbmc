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
    struct S *s = __ESBMC_new_object();
    s->q = &x;
    s->a = n;
    assert(*s->q == 5);
    assert(s->a > 1);
  }
  return 0;
}
