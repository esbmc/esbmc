/* A scalar right operand is promoted to the complex type before the binary
   node is built: the arithmetic node's operand-consistency check rejects an
   operand narrower than its type, and it runs at construction, before
   adjust_complex_arith would promote it. */
int main(void)
{
  _Complex float a = 3.0f + 4.0fi;

  a *= 2.0f;
  __ESBMC_assert(__real__ a == 6.0f, "*= scalar real");
  __ESBMC_assert(__imag__ a == 8.0f, "*= scalar imag");

  a /= 2.0f;
  __ESBMC_assert(__real__ a == 3.0f, "/= scalar real");
  __ESBMC_assert(__imag__ a == 4.0f, "/= scalar imag");

  a += 1.0f;
  __ESBMC_assert(__real__ a == 4.0f, "+= scalar real");
  __ESBMC_assert(__imag__ a == 4.0f, "+= scalar leaves imag");

  return 0;
}
