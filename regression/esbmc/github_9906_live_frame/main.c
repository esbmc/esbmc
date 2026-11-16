// #9906: the jmp_buf's setjmp is still active in main, so no violation
#include <assert.h>
#include <setjmp.h>
jmp_buf b;
int g;
void f(void)
{
  g = 1;
  longjmp(b, 2);
}
int main(void)
{
  int r = setjmp(b);
  if (r == 0)
    f();
  assert(r == 2 && g == 1);
  return 0;
}
