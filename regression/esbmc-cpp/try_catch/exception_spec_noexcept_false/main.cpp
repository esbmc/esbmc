#include <cassert>

// noexcept(false) declares a *potentially throwing* function: its
// specification is NOT restrictive, so an exception may escape it and be
// caught by the caller without invoking std::terminate.
void f() noexcept(false)
{
  throw 1;
}

int main()
{
  int caught = 0;
  try
  {
    f();
  }
  catch (int)
  {
    caught = 1;
  }
  assert(caught == 1);
  return 0;
}
