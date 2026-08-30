#include <type_traits>
#include <cassert>

template <class T>
int width()
{
  return sizeof(std::remove_all_extents_t<T>);
}

int main()
{
  // remove_all_extents strips every extent, so this is sizeof(char), not the
  // 20 bytes of the whole array.
  assert(width<char[4][5]>() == 20);
  return 0;
}
