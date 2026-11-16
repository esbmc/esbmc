#include <assert.h>

extern _Bool __VERIFIER_nondet_bool(void);

static unsigned x;

static void bump(void)
{
  x++;
}

int main(void)
{
  void (*f)(void) = bump;
  while (__VERIFIER_nondet_bool())
    f();
  assert(x == 0);
  return 0;
}
