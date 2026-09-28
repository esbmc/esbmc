#include <pthread.h>
#include <assert.h>

int g = 0;

struct S
{
  long pad;
  int *v[2];
};

struct S s = {0, {0, &g}};
int **pp = &s.v[1];

// R33: `&s.v[1]` composes a nonzero member offset with a nonzero element
// offset. get_reference_set_rec's index arm used to add the element offset only
// when the base offset was zero and otherwise abandon it, so the descriptor
// reached R31's walk with no offset to spell back out and the race was pruned.
// The two halves in isolation (_addrof_offset, _array_decay) always worked;
// only their composition failed.
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
