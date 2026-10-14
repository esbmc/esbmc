// Negative companion to github_6300_aggregate_throw: the aggregate exception is
// caught and carries v == 9, so asserting a wrong value is violated.
#include <cassert>

struct E { int v; };

int main()
{
  try { throw E{9}; }
  catch (E e) { assert(e.v == 42); } // wrong on purpose: v is 9
  return 0;
}
