// std::bad_exception substitution ([except.unexpected]). f's dynamic spec
// throw(std::bad_exception) is violated by `throw 42` (int). That runs the
// installed unexpected handler, which throws a double — also not allowed. Since
// the spec lists std::bad_exception, the in-flight exception is replaced by a
// std::bad_exception, which the spec permits, so it propagates and is caught by
// catch(std::bad_exception&). The lowering models this substitution; the
// imperative path does not (it reports a spurious specification violation), so
// this is a lowered-path-only test.
#include <exception>

void my_unexp()
{
  throw 3.14; // double: not allowed by f's spec
}

void f() throw(std::bad_exception)
{
  throw 42; // int: not allowed -> std::unexpected
}

int main()
{
  std::set_unexpected(my_unexp);
  try
  {
    f();
  }
  catch (std::bad_exception &)
  {
    return 0; // the substituted bad_exception lands here
  }
  catch (...)
  {
    return 2;
  }
  return 1;
}
