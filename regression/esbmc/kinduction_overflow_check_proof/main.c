// The inductive step proves an unbounded loop free of overflow.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int i = 0;
  while (__VERIFIER_nondet_int())
  {
    if (i < 100)
      i = i + 1;
  }
  return 0;
}
