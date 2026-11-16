// #9906: f has returned, so its setjmp is no longer active
#include <assert.h>
#include <setjmp.h>
jmp_buf b;
void f(void)
{
  setjmp(b);
}
int main(void)
{
  f();
  longjmp(b, 1);
  assert(0);
  return 0;
}
