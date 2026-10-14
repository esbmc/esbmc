// Soundness regression: the inner loop terminates (counts j to 10)
// but the outer never modifies p, so the outer is non-terminating
// when p > 0. The certifier MUST NOT report VERIFICATION SUCCESSFUL
// here — neither the inner summary's havoc nor the inner exit
// guard touches p, so no measure on p strictly decreases and
// `prove_loop_terminates` for the outer must fail.
//
// Pins the contract that summarising an inner loop only havocs
// symbols the inner actually modifies; it does NOT havoc the
// outer's measure variable.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int p = __VERIFIER_nondet_int();
  if (p <= 0)
    return 0;
  while (p > 0)
  {
    int j = 0;
    while (j < 10)
      j = j + 1;
    // p is intentionally never decremented — outer is non-terminating.
  }
  return 0;
}
