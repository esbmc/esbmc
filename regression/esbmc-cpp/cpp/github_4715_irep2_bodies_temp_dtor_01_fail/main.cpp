// __ESBMC_assert is a built-in intrinsic; no include needed.

// Failing companion to github_4715_irep2_bodies_temp_dtor_01: with the fix the
// temporary object's destructor runs (dtor_calls == 1), so the assertion below
// is violated and verification FAILS. Without the fix the destructor was
// dropped (dtor_calls == 0) and this assertion would spuriously hold -- so the
// two tests pin the fix from both directions.
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
  __ESBMC_assert(dtor_calls == 0, "dtor count");
  return 0;
}
