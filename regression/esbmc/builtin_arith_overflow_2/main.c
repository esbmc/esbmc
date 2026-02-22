typedef unsigned long ulong;
extern ulong nondet_ulong();

#include <limits.h>
#include <assert.h>

int main()
{
  ulong content_len = nondet_ulong();
  ulong digit = nondet_ulong() % 10; // Simulate a single digit (0-9)
  int overflow_detected = (content_len > (ULONG_MAX - digit) / 10UL);
  int overflow_detected_builtin = 0;
  // First argument will be overwritten with the result if no overflow occurs
  if (__builtin_umull_overflow(content_len, 10UL, &content_len))
  {
    // Overflow detected by built-in function
    overflow_detected_builtin = 1;
  }
  else if (__builtin_uaddl_overflow(content_len, digit, &content_len))
  {
    // Overflow detected by built-in function
    overflow_detected_builtin = 1;
  }
  assert(
    overflow_detected == overflow_detected_builtin &&
    "Overflow detection mismatch");
}