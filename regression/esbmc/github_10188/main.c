// #10188: the NULL check on a pointer with an unknown value set holds once the
// program has ruled NULL out.
extern void *__VERIFIER_nondet_pointer(void);
int x;

int main(void)
{
  int *d = __VERIFIER_nondet_pointer();
  if (d != &x)
    return 0;
  return *d;
}
