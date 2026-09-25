// shared_ptr<void> and unique_ptr<void, D> must instantiate: operator*
// returns add_lvalue_reference<T>::type, which is void here.
#include <cassert>
#include <memory>
struct D { void operator()(void *) const {} };
int main() {
  std::shared_ptr<void> p;
  std::shared_ptr<const void> q = std::make_shared<int>(3);
  std::unique_ptr<void, D> u;
  assert(!q);
  return 0;
}
