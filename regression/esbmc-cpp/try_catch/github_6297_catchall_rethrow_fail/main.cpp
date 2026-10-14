// Negative companion to github_6297_catchall_rethrow: the rethrow inside
// catch(...) preserves the value (7), so asserting a wrong value is violated.
#include <cassert>

int f()
{
  try { throw 7; }
  catch (...) { throw; }
}

int main()
{
  try { f(); }
  catch (int e) { assert(e == 42); } // wrong on purpose: the value is 7
  return 0;
}
