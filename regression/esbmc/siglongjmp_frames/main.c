#include <assert.h>
#include <setjmp.h>

sigjmp_buf outer;
jmp_buf rb;
int depth, trail;

void leave(int v)
{
  trail = trail * 10 + 1;
  siglongjmp(outer, v);
  trail = -1;
}

int nest(void)
{
  if (++depth < 3)
    return nest();
  leave(0);
  return -1;
}

int f(int n)
{
  if (n < 0)
    return -1;
  if (n == 2)
    if (_setjmp(rb) == 5)
      return n;
  if (n == 0)
    _longjmp(rb, 5);
  return f(n - 1);
}

int main(void)
{
  switch (sigsetjmp(outer, 1))
  {
  case 0:
    nest();
    assert(0);
  case 1:
    break;
  default:
    assert(0);
  }
  assert(depth == 3 && trail == 1);
  assert(f(3) == 2);
  return 0;
}
