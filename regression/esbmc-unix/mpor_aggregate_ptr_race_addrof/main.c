#include <pthread.h>
#include <assert.h>

int g = 0;

struct S
{
  int *p;
};

struct S s = {&g};
int **pp = &s.p;

// R31: `&s.p` refers to the struct symbol with the member erased into a byte
// offset, so resolving `**pp` asked the value set for `s`, which nothing keys,
// and MPOR pruned the race. get_value_set_rec now spells the offset back out
// as a field path. Member offset 0 here; see _addrof_offset for a nonzero one.
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
