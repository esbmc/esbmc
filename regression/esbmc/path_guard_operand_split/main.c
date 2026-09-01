/* A caller and callee both derive the same nibbles from one operand
 * byte; the caller constrains the encoding to the illegal set before
 * the call, so the callee's own range check is already decided by the
 * path condition — but matching the two requires chasing the split
 * m = (mn >> 4) & 0x0F as a stable value, not just bare copies. */
unsigned char nondet_u1(void);

static int handler(unsigned char mn)
{
  unsigned char m = (mn >> 4) & 0x0F;
  unsigned char n = mn & 0x0F;
  if (m < 1 || m > 2 || n < 1 || n > 2)
    return -1;
  int acc = 0;
  for (unsigned char i = 0; i < (unsigned char)(m * 8u + n); i++)
    acc++;
  return acc;
}

int main(void)
{
  unsigned char mn = nondet_u1();
  unsigned char m_ = (mn >> 4) & 0x0F, n_ = mn & 0x0F;
  if (m_ >= 1 && m_ <= 2 && n_ >= 1 && n_ <= 2)
    return 0;
  __ESBMC_assert(handler(mn) == -1, "illegal encodings refuse");
  return 0;
}
