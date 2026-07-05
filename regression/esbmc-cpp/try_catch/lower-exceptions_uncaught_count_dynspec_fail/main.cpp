// Negative companion to lower-exceptions_uncaught_count_dynspec: after the
// std::unexpected replacement throw (X) is caught, the count is 0 — the original
// A is replaced, not added. Asserting an exception is still uncaught here must
// fail, which also proves the count is exactly 0 (not an over-count of 1).
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
  throw X();
}

void f() throw(X)
{
  throw A();
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
    assert(std::uncaught_exception()); // wrong: the count is 0 here
  }
  return 0;
}
