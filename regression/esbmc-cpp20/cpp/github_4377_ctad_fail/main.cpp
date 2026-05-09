#include <cassert>

template <typename T>
struct Box
{
  T t;
};

int main()
{
  Box b{42};
  // CTAD-deduced Box<int>; b.t == 42, so the assertion below must fail.
  assert(b.t == 99);
  return 0;
}
