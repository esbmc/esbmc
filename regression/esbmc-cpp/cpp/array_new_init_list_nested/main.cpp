// A nested braced list: every row is stored, a short row and the missing
// rows are value-initialised.
#include <cassert>
int main()
{
  int(*p)[2] = new int[3][2]{{1, 2}, {3}};
  assert(p[0][0] == 1 && p[0][1] == 2 && p[1][0] == 3);
  assert(p[1][1] == 0 && p[2][0] == 0 && p[2][1] == 0);
  delete[] p;
  return 0;
}
