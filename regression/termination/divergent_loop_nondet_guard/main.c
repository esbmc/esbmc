/* `while (nondet())` can keep choosing to iterate whatever x is, so reaching
 * the loop is a non-terminating execution. */
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = 0;
  while (__VERIFIER_nondet_int())
    x++;
  return x;
}
