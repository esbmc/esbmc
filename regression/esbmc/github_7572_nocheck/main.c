/* --no-fp-conversion-check drops only the conversion claim; the rest of
   --overflow-check is untouched. SV-COMP's no-overflow property is signed
   integer arithmetic only ("conversions to signed-integer types do not violate
   this property"), and esbmc-wrapper.py reports any violation in an overflow
   run as FALSE_OVERFLOW, so the competition passes this flag. */
int main(void)
{
  double d = 1e300;
  long long x = (long long)d;

  int a = 2147483647;
  int b = a + 1;
  return (int)x + b;
}
