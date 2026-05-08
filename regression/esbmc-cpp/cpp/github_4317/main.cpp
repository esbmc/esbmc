#include <cassert>
struct S
{
  int x;
};
int main()
{
  S cases[][2] = {{{1}, {2}}};
  for (auto &row : cases)
    assert(row[0].x == 1);
  return 0;
}
