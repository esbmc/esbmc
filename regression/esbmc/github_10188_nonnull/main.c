// #10188: a nondet pointer that passed a NULL test fails no NULL claim.
extern void *__VERIFIER_nondet_pointer(void);
int main(void)
{
  int *d = __VERIFIER_nondet_pointer();
  if (d)
    return *d;
  return 0;
}
