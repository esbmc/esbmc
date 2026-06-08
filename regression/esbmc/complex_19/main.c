int main()
{
  __complex__ double z;
  __real__ z = 3.0;
  __imag__ z = 4.0;

  _Bool b = (_Bool)z;

  assert(b == 1);

  return 0;
}
