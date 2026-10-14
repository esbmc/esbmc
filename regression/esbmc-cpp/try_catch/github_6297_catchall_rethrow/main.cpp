// github #6297: a bare `throw;` inside a catch(...) handler must re-raise the
// in-flight exception so an enclosing handler catches it. The catch-all landing
// previously pushed the handled exception *after* the rethrow, so the re-raise
// found an empty handled stack. Class exceptions use a constructor here (the
// aggregate-init path has a separate typeid issue, see the sibling report).
#include <cassert>

int f()
{
  try { throw 7; }
  catch (...) { throw; }
}

struct E { int v; E(int x) : v(x) {} };

int main()
{
  // scalar, rethrow across a function boundary
  try { f(); }
  catch (int e) { assert(e == 7); }

  // class exception, rethrow re-selects a typed handler
  try { try { throw E(9); } catch (...) { throw; } }
  catch (E e) { assert(e.v == 9); }

  // nested catch-all rethrows
  try { try { try { throw 3; } catch (...) { throw; } } catch (...) { throw; } }
  catch (int e) { assert(e == 3); }
  return 0;
}
