// #10749: the nondet call in pthread_create's arguments returns before the spawn.
#include <pthread.h>
#include <assert.h>
extern int __VERIFIER_nondet_int(void);
void *t(void *a)
{
  assert((long)a != 5);
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, (void *)(long)__VERIFIER_nondet_int());
  return 0;
}
