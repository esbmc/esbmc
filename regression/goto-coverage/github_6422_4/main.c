// No asserts to instrument, so this exercises the remaining_claims == 0
// early return in bmct::run_thread, never reaching multi_property_check.
// Control case for the --schedule fix, not a regression pin for it
// (see github_6422_5 for that).
int main(void)
{
  int x = __VERIFIER_nondet_int();
  if (x > 0)
    return 1;
  return 0;
}
