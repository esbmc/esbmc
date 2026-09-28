/* Soundness twin: the caller does NOT constrain the encoding, so the
 * callee's range check is genuinely open — legal encodings reach the
 * accumulation and refute the blanket refusal claim. Over-eager
 * operand-split matching would prove this vacuously. */
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
  __ESBMC_assert(handler(mn) == -1, "legal encodings must stay reachable");
  return 0;
}
