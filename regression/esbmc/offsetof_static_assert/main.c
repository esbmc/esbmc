#include <stddef.h>
#include <assert.h>

/* offsetof must be an integral constant expression: usable in _Static_assert
 * and in static initialisers, not only in runtime arithmetic. */
struct plain
{
  int a;
  int b;
};

struct nested
{
  int a;
  union
  {
    int u1;
    struct
    {
      int s1;
    };
  };
};

_Static_assert(offsetof(struct plain, b) == 4, "plain member");
_Static_assert(offsetof(struct nested, u1) == 4, "anonymous union member");
_Static_assert(offsetof(struct nested, s1) == 4, "anonymous struct member");

static const unsigned table[] = {
  offsetof(struct plain, a),
  offsetof(struct plain, b),
};

int main(void)
{
  assert(table[0] == 0);
  assert(table[1] == 4);

  /* The dynamic form must still work, via the pointer-arithmetic fallback. */
  struct arr
  {
    int hdr;
    int elems[10];
  };
  unsigned idx = 3;
  assert(offsetof(struct arr, elems[idx]) == 4 + 3 * sizeof(int));
  return 0;
}
