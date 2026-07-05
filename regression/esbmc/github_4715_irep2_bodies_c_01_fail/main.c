// Negative variant for V.4.3 C frontend under --irep2-bodies (esbmc#4715).
// An initialized declaration whose value is verified to be wrong must produce
// VERIFICATION FAILED — exercises that variables survive the round-trip with
// their initializers intact (wrong init → failed assert, not wrong value).

int main()
{
  int x = 10; // initialized decl; round-trip must preserve init value
  __ESBMC_assert(x != 10, "x != 10"); // deliberately false
  return 0;
}
