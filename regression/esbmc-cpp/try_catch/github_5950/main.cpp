#include <stdexcept>

// Issue #5950: merely constructing a std::runtime_error reported a spurious
// use-after-free inside the implicit ~bad_alloc.  The thrown-object temporaries
// of the `throw std::bad_alloc()` in __refcnted_cstr's ctor leaked their
// destructors onto the enclosing scope and ran on the allocation-succeeds path
// where they were never constructed.  With --force-malloc-success the allocation
// cannot fail, so construction must verify cleanly.
int main()
{
  std::runtime_error e("x");
  return 0;
}
