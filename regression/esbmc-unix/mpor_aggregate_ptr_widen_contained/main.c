#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;
int h = 0;

struct S
{
  int *q;
  int *p;
};

struct S arr[2] = {{&h, &g}, {&h, &g}};
int **pp;

// The unknown-offset route over-yields on purpose: resolving `**pp` takes both
// `[].q` and `[].p`, so the value set says the writer may touch `h` when it can
// only touch `g`. That extra target must stay inside the value set -- the
// dereference guards it on a pointer equality that cannot hold. This assertion
// fails the moment it leaks into the encoding, which _symbolic_offset_locked
// cannot detect, since that one passes on unpatched code too.
void *writer(void *arg)
{
  (void)arg;
  **pp = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  pp = &arr[i].p;
  pthread_create(&t, 0, writer, 0);
  h = 5;
  int seen = h;
  pthread_join(t, 0);
  assert(seen == 5);
  return 0;
}
