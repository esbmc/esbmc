#include <assert.h>
#include <pthread.h>

extern int __VERIFIER_nondet_int(void);

int g;

void *t(void *arg)
{
  assert(g != 0);
  return 0;
}

int main()
{
  if (__VERIFIER_nondet_int())
    g = __VERIFIER_nondet_int();
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
