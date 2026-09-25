#include <pthread.h>
#include <assert.h>

int g = 0;

struct S
{
  int *p;
};

struct S s = {&g};
int **pp = &s.p;

// R31 control: the same write with the pointer copied to a local first, which
// MPOR does resolve. Pairs with _addrof so the pair fails if either the defect
// is fixed or the working shape regresses.
void *writer(void *arg)
{
  (void)arg;
  int *lp = *pp;
  *lp = 1;
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
