/* A compound assignment over a complex operand is rebuilt by goto_convert's
   remove_assignment long after adjustment, so the component-level lowering
   never saw it and the SMT layer was handed a raw complex operator. `/=` also
   needed gen_zero to know how to build a zero complex (#6713). */
int main(void)
{
  _Complex float a = 3.0f + 4.0fi;

  a += 1.0f + 2.0fi;
  __ESBMC_assert(__real__ a == 4.0f, "+= real");
  __ESBMC_assert(__imag__ a == 6.0f, "+= imag");

  a -= 1.0f + 2.0fi;
  __ESBMC_assert(__real__ a == 3.0f, "-= real");
  __ESBMC_assert(__imag__ a == 4.0f, "-= imag");

  a *= 2.0f;
  __ESBMC_assert(__real__ a == 6.0f, "*= real");
  __ESBMC_assert(__imag__ a == 8.0f, "*= imag");

  a /= 2.0f;
  __ESBMC_assert(__real__ a == 3.0f, "/= real");
  __ESBMC_assert(__imag__ a == 4.0f, "/= imag");

  _Complex float b = 4.0f;
  b /= 2.0f;
  __ESBMC_assert(__real__ b == 2.0f, "scalar /= real");
  __ESBMC_assert(__imag__ b == 0.0f, "scalar /= imag");

  return 0;
}
