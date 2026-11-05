/* Under fair scheduling the other thread ends the program, so every
 * execution terminates. The divergence check skips programs that create
 * threads, so reaching the loop is not reported, but termination is not
 * proved either. */
#include <pthread.h>
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);

void *quit(void *arg)
{
  exit(0);
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, quit, 0);
  while (__VERIFIER_nondet_int())
    ;
  return 0;
}
