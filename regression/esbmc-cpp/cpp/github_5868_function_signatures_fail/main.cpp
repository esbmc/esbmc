#include <functional>
#include <cassert>

int main()
{
  std::function<double(double)> f = [](double x) { return x + 0.5; };
  // The argument and result are doubles, so this is 2.5 -- routing the call
  // through an int would truncate it to 2.
  assert(f(2.0) == 2.0);
  return 0;
}
