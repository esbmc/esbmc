// Issue #10188: a nondet pointer known to be valid may be dereferenced
extern void *__VERIFIER_nondet_pointer(void);
void __ESBMC_assume(_Bool);
int main(void)
{
  int x = 1;
  int *d = __VERIFIER_nondet_pointer();
  __ESBMC_assume(d == &x);
  return *d - 1;
}
