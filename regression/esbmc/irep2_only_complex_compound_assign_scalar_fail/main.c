/* The scalar promotion carries the real arithmetic, so a wrong expected
   component must be caught rather than excused by an undecomposed node. */
int main(void)
{
  _Complex float a = 3.0f + 4.0fi;

  a *= 2.0f;
  __ESBMC_assert(__imag__ a == 4.0f, "*= scalar imag is 8, not 4");

  return 0;
}
