// #10188: a read through a nondet pointer that is NULL is a NULL dereference.
extern void *__VERIFIER_nondet_pointer(void);
int main(void)
{
  int *d = __VERIFIER_nondet_pointer();
  if (d == 0)
    return *d;
  return 0;
}
