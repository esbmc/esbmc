// std::system_error carries its code through a throw. --force-malloc-success is
// needed because std::runtime_error's message storage allocates, and ESBMC
// models malloc as able to fail: without the flag the escaping std::bad_alloc
// masks the test, and a plain `throw std::runtime_error("x")` fails the same way
// on master, independently of <system_error>.
#include <system_error>
#include <cassert>

int main()
{
  try
  {
    throw std::system_error(std::make_error_code(std::errc::timed_out));
  }
  catch (const std::system_error &ex)
  {
    assert(ex.code().value() == 110);
    assert(ex.code().category() == std::generic_category());
    return 0;
  }
  assert(0); // the handler must run
  return 1;
}
