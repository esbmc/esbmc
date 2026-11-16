/* A do-while loop has no instruction at its `do` to place the witness's
 * transition invariant on, so it gets none. */
extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  do
  {
    x--;
  } while (x > 0);
  return 0;
}
