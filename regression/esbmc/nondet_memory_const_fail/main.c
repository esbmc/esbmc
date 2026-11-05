extern void __VERIFIER_nondet_memory(void *, unsigned long);

/* Modifying a const object is undefined (C11 6.7.3p6), havocked whole or not. */
static const int c = 1;

int main()
{
  __VERIFIER_nondet_memory((void *)&c, sizeof c);
  return 0;
}
