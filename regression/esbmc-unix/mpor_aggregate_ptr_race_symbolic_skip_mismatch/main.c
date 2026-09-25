#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;

struct S
{
  long a;
  double b;
  int *p;
};

struct S arr[2] = {{0, 0.0, &g}, {0, 0.0, &g}};
int **pp;

// R32, past the mismatches: the unset offset makes the descent try every
// member, and the pointer is reached only by walking through two members whose
// type does not match the dereference. A walk that stopped at the first
// non-contributing member would still pass _symbolic_struct_member, whose
// leading member is a pointer of the right type.
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
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
