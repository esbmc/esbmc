// esbmc/esbmc#4715 (V.4.4 parity): negative variant of malloc_01. With the
// allocated object correctly typed `pair` across the --irep2-bodies round-trip,
// the struct copy preserves the fields, so the wrong-value assertion below is a
// genuine violation (q.value is 20, not 21). This guards against the positive
// test passing vacuously (e.g. if the assert were silently dropped).
#include <assert.h>
#include <stdlib.h>

typedef struct
{
  int id;
  void *value;
} pair;

int main()
{
  pair *p = (pair *)malloc(sizeof(pair));
  if (!p)
    return 0;
  p->id = 10;
  p->value = (void *)20;

  pair q = *p; // struct copy through the malloc'd object
  assert(q.id == 10);
  assert(q.value == (void *)21); // wrong: q.value == 20

  free(p);
  return 0;
}
