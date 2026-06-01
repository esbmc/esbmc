#include <stdlib.h>
// Companion to scanf_malloc_bug_3: same malloc'd int buffer, but the scanf
// field width (%9d) fits a 32-bit int, so the input-overflow check must NOT
// fire. Exercises the &arr[i] pointer-arithmetic path in
// goto_checkt::input_overflow_check and guards against over-triggering.
int main(int argc, char *argv[])
{
  int *arr = (int *)malloc(3 * sizeof(int));
  for(int i = 0; i < 3; i++)
  {
    scanf("%9d", &arr[i]);
  }
  return 0;
}
