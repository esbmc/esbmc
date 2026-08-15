#include <assert.h>

int main(void)
{
  /* (4+2i) / (1+1i) = 3 - 1i, not 99: a sign error in the imaginary
     component lands here rather than on the shape of the lowering.
     Spelled with _Complex and __real__/__imag__ because MSVC ships no C99
     complex header. */
  double _Complex a, b;
  __real__ a = 4.0;
  __imag__ a = 2.0;
  __real__ b = 1.0;
  __imag__ b = 1.0;

  double _Complex z = a / b;
  assert(__imag__ z == 99.0);

  return 0;
}
