// Negative companion to github_4715_irep2_bodies_opnew_01: the freshly
// allocated storage is zero, so asserting it equals 1 is violated and
// verification FAILS. Confirms the operator-new round-trip produces a real,
// readable object under --irep2-bodies rather than crashing.
#define NULL 0
int *p;
int main()
{
  p = (int *)operator new(sizeof(int));
  __ESBMC_assert(*p == 1, "zero-initialised storage is not 1");
  return 0;
}
