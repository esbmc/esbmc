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

int main()
{
  // The discarded temporary is destroyed at the end of the full expression,
  // running ~Room then ~Base -> dtor_calls == 2.
  Room();
  __ESBMC_assert(dtor_calls == 2, "dtor count");
  return 0;
}
