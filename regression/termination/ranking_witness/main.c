/* Issue #9656: the ranking-function verdict must write the correctness
 * witness like the forward-condition verdict does. */
extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  while (x > 0)
    x--;
  return 0;
}
