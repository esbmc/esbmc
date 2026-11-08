#include <assert.h>
#include <pthread.h>

extern int __VERIFIER_nondet_int(void);

void *t(void *arg)
{
  int x = __VERIFIER_nondet_int();
  assert(x != 5);
  return 0;
}

int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
