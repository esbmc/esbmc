#include <cassert>
#include <optional>

struct C
{
  static int made;
  C()
  {
    ++made;
  }
};
int C::made = 0;

int main()
{
  std::optional<C> o;
  assert(C::made != 0);
  return 0;
}
