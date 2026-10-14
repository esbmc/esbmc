// --show-vcc prints the VCCs and exits before the probes are ever solved, so
// reached_claims is empty. Both directions of this nondet branch are live, and
// neither may be reported as CWE-561 (issue #4495).
int main(void)
{
  int x = __VERIFIER_nondet_int();
  if (x > 0)
    return 1;
  else
    return 0;
}
