#include <memory>
#include <cstdlib>
#include <cassert>

int main()
{
  int src[2] = {1, 2};
  int *dst = static_cast<int *>(malloc(sizeof(int) * 2));
  if (dst == NULL)
    return 0;

  // [uninitialized.construct]: the _n forms take a count.
  std::uninitialized_copy_n(src, 2, dst);
  assert(dst[0] == 1);
  assert(dst[1] == 2);

  std::uninitialized_fill_n(dst, 2, 7);
  assert(dst[0] == 7);
  assert(dst[1] == 7);
  free(dst);
  return 0;
}
