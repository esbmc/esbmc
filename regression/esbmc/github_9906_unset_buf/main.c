// #9906: b was never set; its zero token must not resume at setjmp(c)
#include <setjmp.h>
jmp_buf b, c;
int main(void)
{
  longjmp(b, 1);
  setjmp(c);
  return 0;
}
