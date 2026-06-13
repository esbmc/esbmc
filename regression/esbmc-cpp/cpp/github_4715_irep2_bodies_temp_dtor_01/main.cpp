#include <cassert>

// Regression for the --irep2-bodies round-trip dropping a temporary object's
// destructor. IREP2 struct types store only data members, so the body
// round-trip strips the inline class type's `methods` and component
// attributes. goto_convert finds a temporary's destructor via get_destructor(),
// which scans the type's methods() and self-compares the destructor's `this`
// type, so a degraded inline type lost the destructor call. get_destructor now
// resolves the inline struct back to its authoritative type symbol via its tag.
//
// Uses a multi-field class on purpose: a single-field/fieldless class
// round-trips to a structurally identical type and would pass even unfixed.
int dtor_calls = 0;

struct Room
{
  int a, b;
  ~Room() { dtor_calls++; }
};

struct House
{
  Room *rm;
  House() { rm = new Room(); }
};

int main()
{
  House h;
  // `new Room()` materialises a temporary that is copied into the heap object
  // and then destroyed, so the destructor runs exactly once during
  // construction. Without the fix it was dropped (dtor_calls == 0).
  assert(dtor_calls == 1);
  return 0;
}
