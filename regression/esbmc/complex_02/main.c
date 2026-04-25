int main()
{
  __complex__ double x;
  __real__ x = 3.0;
  __imag__ x = 4.0;

  assert(__real__ x == 99.0);

  return 0;
}
