int Contains(float value)
{
  for (;;)
    ;
}
float main_min = -0.0f;
int main()
{
  float finite_float;
  float infinite_float;
  float nan_float;

  double finite_double;
  double infinite_double;
  double nan_double;

  char char_1;
  char char_2;
  short short_1;
  short short_2;
  int int_1;
  int int_2;
  long long long_long_1;
  long long long_long_2;
  unsigned char uchar_1;
  unsigned char uchar_2;
  unsigned short ushort_1;
  unsigned short ushort_2;
  unsigned int uint_1;
  unsigned int uint_2;
  unsigned long long ulong_long_1;
  unsigned long long ulong_long_2;

  __ESBMC_assume(finite_float == 1.1234f);
  __ESBMC_assume(infinite_float == 1.0f / 0.0f);
  __ESBMC_assume(nan_float != nan_float);
  __ESBMC_assume(finite_double == 1.1234);
  __ESBMC_assume(infinite_double == 1.0 / 0.0);
  __ESBMC_assume(nan_double != nan_double);
  __ESBMC_assume(char_1 == 'a');
  __ESBMC_assume(char_2 == 0x7f);
  __ESBMC_assume(short_1 == 1);
  __ESBMC_assume(short_2 == 0x7fff);
  __ESBMC_assume(int_1 == 1);
  __ESBMC_assume(int_2 == 0x7fffffff);
  __ESBMC_assume(long_long_1 == 1LL);
  __ESBMC_assume(long_long_2 == 0x7fffffffffffffffLL);
  __ESBMC_assume(uchar_1 == 1U);
  __ESBMC_assume(uchar_2 == 0xffU);
  __ESBMC_assume(ushort_1 == 1U);
  __ESBMC_assume(ushort_2 == 0xffffU);
  __ESBMC_assume(uint_1 == 1U);
  __ESBMC_assume(uint_2 == 0xffffffffU);
  __ESBMC_assume(ulong_long_1 == 1U);
  __ESBMC_assume(ulong_long_2 == 0xffffffffffffffffULL);

  __ESBMC_assume(main_min = finite_float);
  int bar = finite_float + infinite_float + nan_float + finite_double +
            infinite_double + nan_double + char_1 + char_2 + short_1 + short_2 +
            int_1 + int_2 + long_long_1 + long_long_2 + uchar_1 + uchar_2 +
            ushort_1 + ushort_2 + uint_1 + uint_2 + ulong_long_1 + ulong_long_2;
  Contains(bar);
}
