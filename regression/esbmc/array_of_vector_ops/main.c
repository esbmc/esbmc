#include <assert.h>
#include <string.h>
typedef int v4i __attribute__((__vector_size__(16)));
struct S { int tag; v4i v[2]; };
int main(void)
{
  v4i a[3] = {};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      a[i][j] = i * 10 + j;
  assert(a[2][3] == 23 && a[1][0] == 10);
  v4i g[2][2] = {};
  g[1][0][2] = 5;
  assert(g[1][0][2] == 5 && g[0][1][2] == 0);
  unsigned k = 1, m = 2;
  a[k][m] = 99;
  assert(a[1][2] == 99);
  v4i b[3];
  memcpy(b, a, sizeof a);
  assert(b[2][1] == 21 && b[1][2] == 99);
  struct S s = {};
  s.v[1][3] = 7;
  assert(s.v[1][3] == 7 && s.v[0][3] == 0);
  v4i x = a[2];
  assert(x[0] == 20);
  return 0;
}
