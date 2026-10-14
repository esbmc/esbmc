/* The census migrates every adjusted symbol through IREP2 and must leave the
   program alone: a reference and a heap object between them cover the shapes
   whose migration is least trivial. */
#include <cassert>

struct pair
{
  int a;
  int b;
};

int main()
{
  pair p{1, 2};
  int &r = p.a;
  r = 3;
  int *h = new int(4);
  int sum = p.a + p.b + *h;
  delete h;
  assert(sum == 9);
  return 0;
}
