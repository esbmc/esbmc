int main()
{
  __complex__ double x = 3.0 + 2.0i;

  assert(__real__ x == 3.0);
  assert(__imag__ x == 2.0);

  return 0;
}
