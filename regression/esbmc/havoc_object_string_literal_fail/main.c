void __ESBMC_havoc_object(void *);
char c;
int main(void)
{
  char *s = __VERIFIER_nondet_int() ? &c : "lit";
  __ESBMC_havoc_object(s);
  return 0;
}
