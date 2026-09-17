// Each instantiation must get its own closure, so S<int>'s lambda sees
// sizeof(int), not sizeof(char) (#7528).
#include <cassert>

template <class T>
struct S
{
  int g(int x)
  {
    auto b = [&](int i) -> int { return i + x + (int)sizeof(T); };
    return b(1);
  }
};

int main()
{
  S<int> si;
  assert(si.g(5) == 1 + 5 + 1);
  return 0;
}
