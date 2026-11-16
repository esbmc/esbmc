/* The measure reads *p, and \at(p, AnyPrev) would name the old pointer
 * rather than the old value: the witness gets no transition invariant. */
extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int *p = &x;
  while (*p > 0)
    (*p)--;
  return 0;
}
