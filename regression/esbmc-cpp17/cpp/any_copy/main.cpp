// Copying an any used to wrap it instead of copying the held object. The
// converting constructor template was unconstrained, so for a non-const lvalue
// it beat any(const any&) and stored an `any` inside the `any`. [any.cons]/6
// requires that constructor not to participate when decay_t<T> is any itself.
#include <any>
#include <cassert>

int main()
{
  std::any a = 5;
  assert(a.has_value());
  assert(std::any_cast<int>(a) == 5);

  std::any b = a; // copy construction from a non-const lvalue
  assert(b.has_value());
  assert(std::any_cast<int>(b) == 5);

  std::any c;
  assert(!c.has_value());
  c = b; // copy assignment
  assert(c.has_value());
  assert(std::any_cast<int>(c) == 5);

  return 0;
}
