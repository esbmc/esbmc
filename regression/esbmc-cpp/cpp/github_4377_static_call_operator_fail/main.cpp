// esbmc/esbmc#4377 negative control (C++23): a static operator() used to lower its call site as
// if it were a non-static member, passing the object as a hidden first
// argument and clashing with the declared signature.
#include <cassert>

struct F
{
  static int operator()(int x)
  {
    return x + 1;
  }
};

int main()
{
  F f;
  assert(f(2) == 3);
  assert(F{}(41) == 43);
  return 0;
}
