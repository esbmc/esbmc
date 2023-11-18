
#include <setjmp.h>

#undef setjmp
int setjmp(jmp_buf __env)
{
  __ESBMC_unreachable();
}

void longjmp(jmp_buf env, int status)
{
  __ESBMC_unreachable();
}
