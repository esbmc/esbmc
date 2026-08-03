#include <cassert>

int dtor_runs = 0;

struct Guard
{
  ~Guard()
  {
    ++dtor_runs;
  }
};

int side_effect(int x)
{
  return x + 1;
}

// A side-effect return (a call) inside a scope holding a destructor: the
// dispatcher delegates this statement to convert_return while converting the
// rest of the function natively. The destructor must still run exactly once.
int f(int x)
{
  Guard g;
  return side_effect(x);
}

int main()
{
  int r = f(41);
  assert(r == 42);
  assert(dtor_runs == 1);
  return 0;
}
