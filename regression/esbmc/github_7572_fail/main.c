/* C11 6.3.1.4p1: if the integral part of a floating value cannot be represented
   by the destination integer type, the behaviour is undefined. This went
   unchecked, so ESBMC reported SUCCESSFUL on undefined code (#7572). */
int main(void)
{
  double d = 1e300;
  long long x = (long long)d;
  (void)x;
  return 0;
}
