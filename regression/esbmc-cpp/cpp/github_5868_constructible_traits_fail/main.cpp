#include <type_traits>
#include <cassert>

struct no_default
{
  no_default(int)
  {
  }
};

int main()
{
  // no_default has no default constructor, so the trait is false.
  assert(std::is_default_constructible<no_default>::value);
  return 0;
}
