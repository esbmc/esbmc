extern void __VERIFIER_nondet_memory(void *, unsigned long);
extern _Bool __VERIFIER_nondet_bool(void);

int x, y;

int main()
{
  int *p = __VERIFIER_nondet_bool() ? &x : &y;
  __VERIFIER_nondet_memory(p, sizeof x);
  return 0;
}
