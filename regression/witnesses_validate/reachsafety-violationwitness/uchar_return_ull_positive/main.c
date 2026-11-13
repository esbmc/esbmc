extern unsigned char __VERIFIER_nondet_uchar(void);
int main(void)
{
  unsigned char x = __VERIFIER_nondet_uchar();
  __ESBMC_assert(x != 255, "target");
}
