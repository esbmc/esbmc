// Counterpart to lower-exceptions_bad_exception_subst: f's dynamic spec
// throw(int) does NOT list std::bad_exception. `throw 'c'` (char) violates it
// and runs the unexpected handler, which throws a double — also not allowed.
// With no std::bad_exception in the spec there is no substitution, so
// std::terminate is called ([except.unexpected]); the lowering models this as a
// verification failure. Lowered-path-only (the imperative path agrees here, but
// the sibling SUCCESSFUL test only lowers, so keep the pair consistent).
#include <exception>

void my_unexp()
{
  throw 3.14; // double: not allowed
}

void f() throw(int)
{
  throw 'c'; // char: not allowed -> std::unexpected
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
    return 0;
  }
  catch (...)
  {
    return 2;
  }
  return 1;
}
