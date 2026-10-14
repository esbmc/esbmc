#include <type_traits>
#include <cassert>

struct target
{
};
struct explicit_src
{
  explicit operator target() const
  {
    return target();
  }
};

int main()
{
  // [meta.rel] makes this false; asserting the pre-fix answer must fail.
  assert((std::is_convertible<explicit_src, target>::value));
  return 0;
}
