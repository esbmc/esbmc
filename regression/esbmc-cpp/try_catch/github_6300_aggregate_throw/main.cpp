// github #6300: a thrown class exception that is aggregate-initialised
// (throw E{...}, no user-declared constructor) must be caught by a matching
// handler. The throw's operand carried an inline struct type (get_complete_type
// resolves it during InitListExpr conversion), so its exception id did not
// match the catch clause's symbol-typed id and the exception escaped uncaught.
#include <cassert>

struct E { int v; };
struct B { int b; };
struct D : B { int d; };

int main()
{
  try { throw E{9}; }
  catch (E e) { assert(e.v == 9); }

  try { throw E{7}; }
  catch (const E &e) { assert(e.v == 7); }

  // aggregate-initialised derived object caught as its base
  try { throw D{{1}, 2}; }
  catch (B &b) { assert(b.b == 1); }
  return 0;
}
