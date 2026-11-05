/* The loop exits only when the nondet value is 0 and x >= 10, so always
 * choosing a nonzero value keeps it running. */
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = 0;
  while (__VERIFIER_nondet_int() || x < 10)
    x++;
  return 0;
}
