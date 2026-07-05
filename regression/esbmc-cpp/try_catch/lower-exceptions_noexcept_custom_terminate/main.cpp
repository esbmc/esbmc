// An exception escaping a noexcept function calls std::terminate, which runs the
// installed std::set_terminate handler. Here the handler exits cleanly, so a
// running binary terminates without error — and the lowering, routing the
// terminate point through the OM std::terminate(), honours the handler and
// verifies SUCCESSFUL. (With the default handler the escape is reported FAILED;
// see lower-exceptions_noexcept_escape_fail.)
#include <exception>
#include <cstdlib>

void my_terminate()
{
  std::exit(0);
}

void f() noexcept
{
  throw 42; // escapes noexcept -> std::terminate -> my_terminate -> exit(0)
}

int main()
{
  std::set_terminate(my_terminate);
  f();
  return 0;
}
