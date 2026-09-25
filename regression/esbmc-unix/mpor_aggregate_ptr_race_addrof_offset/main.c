#include <pthread.h>
#include <assert.h>

int g = 0;

struct S
{
  int *q;
  int *p;
};

struct S s = {0, &g};
int **pp = &s.p;

// R31, nonzero member offset: the descent has to walk the struct's members to
// find which one the descriptor's byte offset names, rather than stopping at
// the first. Pairs with _addrof, where the member sits at offset 0.
void *writer(void *arg)
{
  (void)arg;
  **pp = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
