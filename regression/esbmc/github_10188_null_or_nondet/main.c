// #10188: NULL is already a target, so the read gets one NULL claim, not two.
extern void *__VERIFIER_nondet_pointer(void);
extern int __VERIFIER_nondet_int(void);
int main(void)
{
  int *d = __VERIFIER_nondet_pointer();
  if (__VERIFIER_nondet_int())
    d = 0;
  return *d;
}
