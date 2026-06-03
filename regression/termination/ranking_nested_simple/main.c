// Pins the ranking certifier's nested-loop support. The outer's
// body contains an inner counting loop; previously the body parser
// bailed on the inner's backwards-goto. The fix recognises an
// inner-loop head whose location_number is in the loop_skip map
// (every other loop of the function) and emits a "summary" block:
// havoc the inner's modified set with fresh symbols, then attach
// the inner's exit guard (`!G_inner`) as a path-cond atom.
//
// Soundness: the top-level driver still proves the inner loop
// independently via `prove_loop_terminates`. If the inner doesn't
// terminate the function as a whole returns UNKNOWN. The summary
// only changes whether the OUTER's recognise_loop succeeds.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int n = __VERIFIER_nondet_int();
  int m = __VERIFIER_nondet_int();
  if (n < 0)
    n = 0;
  if (m < 0)
    m = 0;
  int i = 0;
  while (i < n)
  {
    int j = 0;
    while (j < m)
      j = j + 1;
    i = i + 1;
  }
  return 0;
}
