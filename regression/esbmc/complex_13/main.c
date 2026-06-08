int main()
{
  __complex__ double x;
  __real__ x = 3.0;
  __imag__ x = 4.0;

  __complex__ double y;
  __real__ y = 1.0;
  __imag__ y = 2.0;

  __complex__ double z = x - y;

  assert(__real__ z == 2.0);
  assert(__imag__ z == 2.0);

  return 0;
}
