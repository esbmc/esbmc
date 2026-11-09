/* Counterpart of ranking_transition_invariant: the loop does not terminate,
 * so the witness is a violation witness with no transition invariant. */
extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int y;
  while (x > 0)
  {
    y = 10;
    x = 1;
  }
  return 0;
}
