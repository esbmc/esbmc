#include <type_traits>
#include <cassert>

struct throwing
{
  throwing()
  {
    throw 1;
  }
};

int main()
{
  // Its default constructor throws, so the trait is false.
  assert(std::is_nothrow_default_constructible<throwing>::value);
  return 0;
}
