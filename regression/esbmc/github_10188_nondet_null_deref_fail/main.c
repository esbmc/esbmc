// Issue #10188: NULL dereference through a nondet pointer
extern void *__VERIFIER_nondet_pointer(void);
int main(void)
{
  int *d = __VERIFIER_nondet_pointer();
  if (d == 0)
    return *d;
  return 0;
}
