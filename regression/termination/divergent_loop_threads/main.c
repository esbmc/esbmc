/* The other thread can end the program while the loop runs, so reaching the
 * loop proves nothing. */
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
