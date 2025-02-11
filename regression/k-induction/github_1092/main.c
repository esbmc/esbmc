#define __VERIFIER_assert(cond)                                                \
  do                                                                           \
  {                                                                            \
    if(!(cond))                                                                \
    {                                                                          \
      __ESBMC_assert(0, "");                                                   \
      abort();                                                                 \
    }                                                                          \
  } while(0)

#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);
extern void abort(void);
#include <assert.h>
void reach_error()
{
  assert(0);
}

int main()
{
  int status = 0;

  while(__VERIFIER_nondet_int())
  {
    if(!status)
    {
      status = 1;
    }
    else if(status == 1)
    {
      status = 2;
    }
    else if(status >= 2)
    {
      status = 3;
    }
  }
  __VERIFIER_assert(status != 3);
  return 0;
}