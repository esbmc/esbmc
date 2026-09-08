// The controls for shared_ptr_member_copy: the neighbouring shapes that do
// verify, so the defect stays localised to copying a class that holds a
// shared_ptr rather than to shared_ptr generally.
#include <memory>
#include <cassert>

struct H
{
  std::shared_ptr<int> p;
};

H make_wrapper(const std::shared_ptr<int> &q)
{
  return H{q};
}

std::shared_ptr<int> make_bare()
{
  return std::make_shared<int>(3);
}

int main()
{
  // copying the shared_ptr itself
  std::shared_ptr<int> a = std::make_shared<int>(1);
  std::shared_ptr<int> b = a;
  assert(*b == 1);

  // a wrapper returned by value, rather than copied from an existing one
  std::shared_ptr<int> q = std::make_shared<int>(2);
  H w = make_wrapper(q);
  assert(*w.p == 2);

  // a bare shared_ptr returned by value
  std::shared_ptr<int> r = make_bare();
  assert(*r == 3);
  return 0;
}
