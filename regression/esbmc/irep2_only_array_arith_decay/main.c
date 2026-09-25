/* clang leaves CK_ArrayToPointerDecay implicit, so an array operand of pointer
   arithmetic reaches migration undecayed when the IREP2 pass is the sole
   adjuster. Building add2t/sub2t on it is malformed. */
int main(void)
{
  char a[9];
  char *q = a;
  char *p = a + 1;
  /* Not `long`: it is 32-bit on LLP64, and the narrowing would put a
     cast in the goto text this test matches on. */
  __PTRDIFF_TYPE__ d = a - q;
  return (int)(p[0] + d);
}
