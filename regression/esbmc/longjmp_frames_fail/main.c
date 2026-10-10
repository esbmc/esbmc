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
  if (setjmp(b) == 0)
    f();
  else
    assert(g == 0);
  return 0;
}
