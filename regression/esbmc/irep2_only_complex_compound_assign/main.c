/* `a op= b` over a complex operand becomes `a = a op b` so the component-level
   decomposition can see it. goto_convert's remove_assignment rebuilds the
   compound form long after adjustment, so a node left here reaches the SMT
   layer as a raw complex operator and the backend faults on it (#6713). */
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

  return 0;
}
