
#include <setjmp.h>

#undef setjmp
int setjmp(jmp_buf __env)
{
  __ESBMC_unreachable();
}

// Due to some macro expansion some programs may have the _setjmp instead
int _setjmp(jmp_buf __env)
{
  setjmp(__env);
}


void longjmp(jmp_buf env, int status)
{
  __ESBMC_unreachable();
}
