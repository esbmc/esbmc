// Negative companion: the value carried by the exception_ptr must round-trip
// faithfully. Asserting the wrong value after rethrow must fail.
#include <exception>
#include <cassert>

struct E
{
  int v;
  E(int x) : v(x)
  {
  }
};

int main()
{
  std::exception_ptr p = std::make_exception_ptr(E(42));
  try
  {
    std::rethrow_exception(p);
  }
  catch (E &e)
  {
    assert(e.v == 7); // wrong: the captured value is 42
  }
  return 0;
}
