// #5393: a flexible array member has no elements of its own, so initialising
// it in a constructor's member-initialiser list constructs none.
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
  assert(cnt == 0);
  return 0;
}
