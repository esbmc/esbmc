typedef __complex__ double cdouble;

int main()
{
  __complex__ double z = 1.0 + 2.0i;

  __complex__ double n = -z;
  assert(__real__ n == -1.0);
  assert(__imag__ n == -2.0);

  __complex__ double c = ~z;
  assert(__real__ c == 1.0);
  assert(__imag__ c == -2.0);

  __complex__ double p = +z;
  assert(__real__ p == 1.0 && __imag__ p == 2.0);

  cdouble t = -z;
  assert(__real__ t == -1.0 && __imag__ t == -2.0);

  __complex__ int zi = 3 + 4i;
  __complex__ int ni = -zi;
  __complex__ int ci = ~zi;
  assert(__real__ ni == -3 && __imag__ ni == -4);
  assert(__real__ ci == 3 && __imag__ ci == -4);

  return 0;
}
