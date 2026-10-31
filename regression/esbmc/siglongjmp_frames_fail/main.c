#include <assert.h>
#include <setjmp.h>

int __sigsetjmp(sigjmp_buf, int);

sigjmp_buf b, c;
int g;

void f(void)
{
  g = 1;
  siglongjmp(b, 2);
}

void h(void)
{
  g = 2;
  siglongjmp(c, 0);
}

int main(void)
{
  if (sigsetjmp(b, 0) == 0)
    f();
  else
    assert(g == 0);
  if (__sigsetjmp(c, 1) == 0)
    h();
  else
    assert(g == 1);
  return 0;
}
