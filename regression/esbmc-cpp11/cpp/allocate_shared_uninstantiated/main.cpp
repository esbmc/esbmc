// github #6488: <memory> declared make_shared but not allocate_shared, so
// merely *naming* it made clang parse the `<` as less-than and reject the
// translation unit -- even from a function template that is never
// instantiated, which no harness can work around.
#include <memory>

template <typename T>
std::shared_ptr<T> MakeShared(const std::allocator<T> &alloc)
{
  return std::allocate_shared<T, std::allocator<T>>(alloc);
}

int main()
{
  return 0;
}
