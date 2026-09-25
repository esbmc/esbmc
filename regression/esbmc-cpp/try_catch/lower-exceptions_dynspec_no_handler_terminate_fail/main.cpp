// A violated dynamic exception specification with NO std::set_unexpected
// installed runs the default unexpected handler, which must terminate
// ([except.unexpected]). Terminate here is a spec violation and must be
// reported as FAILED even though a custom std::set_terminate handler ends the
// path (abort, modeled as assume(0)): the lowering owns this terminate point
// and asserts it directly, rather than routing through std::terminate() where
// the custom handler could silently swallow the violation.
#include <exception>
#include <cstdlib>

struct A
{
};

void my_terminate()
{
  abort();
}

void f() throw(int)
{
  throw A(); // A is not permitted by throw(int) -> default std::unexpected
}

int main()
{
  std::set_terminate(my_terminate);
  f();
  return 0;
}
