#include <assert.h>
#include <setjmp.h>

jmp_buf b;
int g;

int main(void)
{
  void (*jump)(jmp_buf, int) = longjmp;
  if (setjmp(b) == 0)
  {
    g = 1;
    jump(b, 2);
  }
  else
    assert(g == 0);
  return 0;
}
