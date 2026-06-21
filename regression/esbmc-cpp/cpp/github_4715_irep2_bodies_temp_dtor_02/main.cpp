// __ESBMC_assert is a built-in intrinsic; no include needed.

// Inheritance variant of github_4715_irep2_bodies_temp_dtor_01: destroying a
// derived temporary runs both the derived and the base destructor. Exercises
// get_destructor() resolving an inline derived class type (whose `methods` and
// `bases` the --irep2-bodies round-trip strips) back to its type symbol.
int dtor_calls = 0;

struct Base
{
  int x;
  ~Base() { dtor_calls++; }
};

struct Room : Base
{
  int y;
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
  // The temporary built for `new Room()` is destroyed after the copy, running
  // ~Room then ~Base -> dtor_calls == 2. Without the fix both were dropped.
  __ESBMC_assert(dtor_calls == 2, "dtor count");
  return 0;
}
