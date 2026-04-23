int main()
{
  __complex__ double z;
  __real__ z = 3.0;
  __imag__ z = 4.0;

  _Bool b = (_Bool)z;

  /* (3+4i) is non-zero, so b should be 1, not 0 */
  assert(b == 0);

  return 0;
}
