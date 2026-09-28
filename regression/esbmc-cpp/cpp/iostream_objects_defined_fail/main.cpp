// Non-vacuity guard for iostream_objects_defined: a freshly constructed stream
// really is good, so asserting the opposite must FAIL. Before the fix this
// failed too -- but so did its positive counterpart, the tell-tale of a nondet
// read from an undefined object.
#include <iostream>
#include <cassert>

int main()
{
  assert(!std::cout.good());
  return 0;
}
