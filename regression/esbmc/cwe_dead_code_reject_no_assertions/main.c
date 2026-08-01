// The branch probes inserted by goto_coveraget::insert_assert are marked
// user_provided(true), which is exactly what goto_symext::symex_assert drops
// under --no-assertions. That leaves reached_claims empty while all_claims
// still holds every probe, so both branches below would be reported as
// CWE-561 (issue #4495). The combination must be rejected up front.
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  if (x > 0)
    return 1;
  return 0;
}
