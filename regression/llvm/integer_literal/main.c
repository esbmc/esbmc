//#include <stdio.h>
//#include <assert.h>

int main(void)
{
  printf("123 = %d\n", 123);
  printf("0123 = %d\n", 0123);
  printf("0x123 = %d\n", 0x123);
  printf("12345678901234567890ull = %llu\n", 12345678901234567890ull);

  // the type is unsigned long long even without a long long suffix
  printf("12345678901234567890u = %llu\n", 12345678901234567890u );
 
  //  printf("%lld\n", -9223372036854775808); // ERROR
  // the value 9223372036854775808 cannot fit in signed long long, which is the
  // biggest type allowed for unsuffixed decimal integer constant
 
  printf("%llu\n", -9223372036854775808u );
  // unary minus applied to unsigned value subtracts it from 2^64,
  // this gives 9223372036854775808

  printf("%lld\n", -9223372036854775807 - 1);
  // correct way to represent the value -9223372036854775808

  int d = 42;
  int o = 052;
  int x = 0x2a;
  int X = 0X2A;

  assert(d == o);
  assert(d == x);
  assert(d == X);
}
