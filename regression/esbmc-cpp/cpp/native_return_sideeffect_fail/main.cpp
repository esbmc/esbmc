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

int f(int x)
{
  Guard g;
  return side_effect(x);
}

int main()
{
  int r = f(41);
  // The destructor runs exactly once, so this must fail rather than be
  // vacuously true: it pins that the delegated return does not skip or
  // duplicate the scope-exit unwind.
  assert(dtor_runs == 2);
  return r - 42;
}
