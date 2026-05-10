#include <cassert>

struct S
{
  int v = 0;
  void inc(this S &self) { self.v++; }
};

int main()
{
  S s;
  s.inc();
  s.inc();
  // The non-const explicit object reference must mutate the underlying
  // object.
  assert(s.v == 2);
  return 0;
}
