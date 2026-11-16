extern int __VERIFIER_nondet_int(void);
int value;
int main(void)
{
  value = __VERIFIER_nondet_int();
  __ESBMC_assert(value != 1, "target");
}
