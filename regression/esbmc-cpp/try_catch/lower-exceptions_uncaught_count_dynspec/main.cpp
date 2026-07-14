// uncaught-count correctness across dynamic-exception-specification recovery.
// f() has a throw(X) spec but throws A; the violated spec runs the installed
// std::unexpected handler, which throws X (a permitted type). That replacement
// throw propagates and is caught. The original A is *replaced* by X, so exactly
// one exception is uncaught during recovery — the lowering must not double-count
// it. Once X enters its handler the count is back to 0 ([except.unexpected],
// [except.uncaught]). Dynamic specs are pre-C++17, so this uses the singular
// std::uncaught_exception().
#include <exception>
#include <cassert>

struct X
{
};
struct A
{
};

void my_unexpected()
{
  throw X(); // replacement throw, permitted by f's spec
}

void f() throw(X)
{
  throw A(); // violates throw(X): runs unexpected -> my_unexpected throws X
}

int main()
{
  std::set_unexpected(my_unexpected);
  try
  {
    f();
  }
  catch (X &)
  {
    assert(!std::uncaught_exception()); // X is being handled -> count 0
  }
  return 0;
}
