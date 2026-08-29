#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;

struct S
{
  int *p;
};

struct S s = {&g};
struct S t = {&g};
int **pp;

// R31, two descriptors and no constant to fold: `pp` merges both arms, so
// folding `*(&s.p)` at the dereference -- the narrow fix the other tests here
// would all have accepted -- cannot reach either member. The offset has to be
// walked back into a field path per descriptor.
void *writer(void *arg)
{
  (void)arg;
  **pp = 1;
  return 0;
}

int main(void)
{
  pthread_t th;
  int c = nondet_int();
  pp = c ? &s.p : &t.p;
  pthread_create(&th, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(th, 0);
  assert(seen == 2);
  return 0;
}
