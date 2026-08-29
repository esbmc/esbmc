// Indexing through a pointer-to-pointer lowers to a dereference rather than an
// index chain, so the walk finds no enclosing row to linearise and leaves the
// address alone (#6778).
#include <assert.h>

int main(void)
{
  int r0[3] = {0, 1, 2};
  int r1[3] = {3, 4, 5};
  int *rows[2] = {r0, r1};
  int **pp = rows;

  assert(&pp[1][2] == &r1[2]);
  return 0;
}
