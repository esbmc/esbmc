// #10188: a read through a nondet pointer that is NULL is a NULL dereference.
extern void *__VERIFIER_nondet_pointer(void);
int x;
int main(void)
{
  int *d = __VERIFIER_nondet_pointer();
  __ESBMC_assume(d == 0 || d == &x);
  if (d != 0)
    return *d;
  return 0;
}
