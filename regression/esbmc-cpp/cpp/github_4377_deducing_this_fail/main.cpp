// esbmc/esbmc#4377 negative control (C++23): an explicit object parameter used to lower the
// body's member access to a misaligned dereference, giving a spurious
// counterexample on a deterministic program.
#include <cassert>

struct S
{
  int v = 7;
  int get(this S const &self)
  {
    return self.v;
  }
  int doubled(this S const &self)
  {
    return self.v * 2;
  }
};

int main()
{
  S s;
  assert(s.get() == 7);
  assert(s.doubled() == 15);
  s.v = 21;
  assert(s.get() == 21);
  return 0;
}
