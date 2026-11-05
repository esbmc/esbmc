extern void __VERIFIER_nondet_memory(void *, unsigned long);
extern _Bool __VERIFIER_nondet_bool(void);

/* Havocking through a pointer that may be NULL writes through NULL. */
int x;

int main()
{
  int *p = __VERIFIER_nondet_bool() ? &x : (int *)0;
  __VERIFIER_nondet_memory(p, sizeof x);
  return 0;
}
