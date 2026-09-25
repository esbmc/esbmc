#include <type_traits>
#include <memory>
#include <cstdlib>
#include <cassert>

struct plain
{
  int a;
};
struct virt
{
  virtual ~virt()
  {
  }
};

int main()
{
  // [meta.unary.prop]
  static_assert(std::has_virtual_destructor<virt>::value, "virtual dtor");
  static_assert(!std::has_virtual_destructor<plain>::value, "plain");
  static_assert(!std::has_virtual_destructor<int>::value, "int");
  static_assert(std::has_virtual_destructor_v<virt>, "_v alias");

  // [uninitialized.construct]
  int src[2] = {1, 2};
  int *dst = static_cast<int *>(malloc(sizeof(int) * 2));
  if (dst == NULL)
    return 0;
  std::uninitialized_move(src, src + 2, dst);
  assert(dst[0] == 1);
  assert(dst[1] == 2);
  free(dst);
  return 0;
}
