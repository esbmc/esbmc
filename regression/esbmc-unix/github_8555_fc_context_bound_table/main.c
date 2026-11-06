#include <assert.h>
#include <pthread.h>

void __VERIFIER_atomic_begin(void);
void __VERIFIER_atomic_end(void);

int i = 3, j = 6;

void *t1(void *arg)
{
  for (int k = 0; k < 2; ++k)
  {
    __VERIFIER_atomic_begin();
    i = j + 1;
    __VERIFIER_atomic_end();
  }
  return NULL;
}

void *t2(void *arg)
{
  for (int k = 0; k < 2; ++k)
  {
    __VERIFIER_atomic_begin();
    j = i + 1;
    __VERIFIER_atomic_end();
  }
  return NULL;
}

int main(void)
{
  pthread_t id1, id2;
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  __VERIFIER_atomic_begin();
  int cond_i = i >= 10;
  __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin();
  int cond_j = j >= 10;
  __VERIFIER_atomic_end();
  assert(!(cond_i || cond_j));
  return 0;
}
