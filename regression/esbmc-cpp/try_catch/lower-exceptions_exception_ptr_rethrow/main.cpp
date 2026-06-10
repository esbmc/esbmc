// std::current_exception() captures the currently handled exception and
// std::rethrow_exception() makes it propagate again. The lowered path should
// preserve the original object across the capture and rethrow.
#include <exception>
#include <cassert>

struct X
{
  int v;
  X(int n) : v(n)
  {
  }
};

int main()
{
  std::exception_ptr ep;

  try
  {
    throw X(7);
  }
  catch (...)
  {
    ep = std::current_exception();
    assert((bool)ep);
  }

  try
  {
    std::rethrow_exception(ep);
    assert(false);
  }
  catch (X &x)
  {
    assert(x.v == 7);
  }

  return 0;
}
