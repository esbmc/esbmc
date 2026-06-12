// esbmc/esbmc#4715 (V.4.4 parity): under --irep2-bodies the body round-trip
// dropped the folded sizeof(T) element type (#c_sizeof_type) from the malloc
// size constant, so malloc(sizeof(pair)) allocated a `char` blob instead of a
// `pair`. The subsequent struct copy `q = *p` then lost the field layout and
// the value assertions failed spuriously. constant_int2t now carries the
// sizeof type across the round-trip, so the allocated object is typed `pair`.
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
  assert(q.value == (void *)20);

  free(p);
  return 0;
}
