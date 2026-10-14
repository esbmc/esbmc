// #7797: layout_right varies the last index fastest, so the leading stride is
// the product of the extents after it, not 1. [mdspan.layout.right]
#include <mdspan>
#include <cassert>
#include <cstddef>

int main()
{
  int buf[6] = {0, 1, 2, 3, 4, 5};
  std::mdspan<int, std::extents<size_t, 2, 3>> m(buf);
  assert(m.stride(0) == 1);
  return 0;
}
