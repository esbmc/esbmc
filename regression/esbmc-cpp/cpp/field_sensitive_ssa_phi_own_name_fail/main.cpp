#include <cassert>
#include <cstdlib>

struct iter
{
  int *key;
  int pos;
  iter operator++(int)
  {
    pos++;
    int v = key[pos];
    (void)v;
    return *this;
  }
};

struct box
{
  int *k;
  int size;
  iter begin()
  {
    iter t;
    t.key = k;
    t.pos = 0;
    return t;
  }
  iter find(int x)
  {
    iter it = begin();
    for (int i = 0; i < size; i++)
    {
      if (k[i] == x)
      {
        it++;
        return it;
      }
      it++;
    }
    return begin();
  }
};

int main()
{
  box b;
  b.k = (int *)malloc(4 * sizeof(int));
  if (!b.k)
    return 0;
  b.size = 4;
  b.k[0] = 1;
  b.k[1] = 2;
  iter r = b.find(9);
  assert(r.pos == 0);
}
