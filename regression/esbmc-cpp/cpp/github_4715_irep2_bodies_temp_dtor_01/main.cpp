// __ESBMC_assert is a built-in intrinsic; no include needed.

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

int main()
{
  // The discarded temporary is destroyed at the end of the full expression,
  // running ~Room exactly once. If get_destructor() fails on the degraded
  // inline type, no destructor is scheduled and the count stays 0.
  Room();
  __ESBMC_assert(dtor_calls == 1, "dtor count");
  return 0;
}
