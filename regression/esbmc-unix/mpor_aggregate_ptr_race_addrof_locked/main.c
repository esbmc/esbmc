#include <pthread.h>
#include <assert.h>

int g = 0;

struct S
{
  int *p;
};

struct S s = {&g};
int **pp = &s.p;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

// The passing direction of the R31 fix: the same aggregate-held pointer and the
// same two writers, but mutually excluded, so `seen == 2` holds on every
// interleaving. Resolving `**pp` now reaches `g`, which is what lets MPOR see
// the two critical sections at all -- this test fails if that added reach ever
// turns a correct program into a false alarm. It is the only test here that
// exercises the new descent and expects SUCCESSFUL.
void *writer(void *arg)
{
  (void)arg;
  pthread_mutex_lock(&m);
  **pp = 1;
  pthread_mutex_unlock(&m);
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  pthread_mutex_lock(&m);
  g = 2;
  int seen = g;
  pthread_mutex_unlock(&m);
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
