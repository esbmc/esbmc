#include <cassert>

template <typename>
struct a;
a<char> *b;
struct c;
template <typename>
struct a
{
  c *b;
};
struct c : a<char>
{
};

int main()
{
  b->b = 0;
  assert(b->b == 0);
}
