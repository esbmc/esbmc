/* Test #pragma unroll with undefined macro
 * Should fail with parsing error since UNDEFINED_MACRO is not defined
 */

int main() {
  int sum = 0;

  #pragma unroll UNDEFINED_MACRO
  for (int i = 0; i < 10; i++) {
    sum += i;
  }

  return 0;
}
