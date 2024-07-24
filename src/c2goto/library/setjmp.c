
#include <setjmp.h>

#undef setjmp

int setjmp(jmp_buf __env)
{
  __ESBMC_unreachable();
  return nondet_int();
}

// Due to some macro expansion some programs may have the _setjmp instead
#ifdef __MINGW32__
int __cdecl __attribute__((__nothrow__, __returns_twice__))
_setjmp(jmp_buf __env, void *_Ctx)
#else
int _setjmp(jmp_buf __env)
#endif
{
  return setjmp(__env);
}

void longjmp(jmp_buf env, int status)
{
  __ESBMC_unreachable();
}
