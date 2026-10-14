// A flexible array member sized by the calloc that allocated it: the three ints
// paid for are in bounds, and reading them back must not be flagged. Pins the
// other half of github_133_flex_array_fail -- a bounds check that fired on
// everything would pass that test while breaking this one. #133
#include <stdlib.h>

struct A
{
  char alloc;
  int b[];
};

int main()
{
  struct A *e = calloc(1, sizeof(struct A) + 3 * sizeof(int));
  if (!e)
    return 0;

  e->b[0] = 1;
  e->b[1] = 2;
  e->b[2] = 3;
  int s = e->b[0] + e->b[1] + e->b[2];

  free(e);
  return s == 6 ? 0 : 1;
}
