// Counterpart of github_5393_meminit: no element is constructed, so asserting
// that one was is violated.
#include <cassert>

int cnt;

struct E
{
  E()
  {
    cnt++;
  }
};

struct T
{
  int n;
  E fam[];
  T() : n(0), fam()
  {
  }
};

int main()
{
  T t;
  assert(cnt == 1);
  return 0;
}
