#include <stdarg.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

int
add_em_up (int count,...)
{
  va_list ap;
  int i, sum;

  va_start (ap, count);

  sum = 0;
  for (i = 0; i < count; i++)
    sum += va_arg (ap, int);

  va_end (ap);

  return sum;
}

double stddev(int count, ...) 
{
    double sum = 0;
    double sum_sq = 0;
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; ++i) {
        double num = va_arg(args, double);
        sum += num;
        sum_sq += num*num;
    }
    va_end(args);
    return sqrt(sum_sq/count - (sum/count)*(sum/count));
}

void simple_printf(const char* fmt,...)
{
    va_list args;
    va_start(args, fmt);
 
    while (*fmt != '\0') {
        if (*fmt == 'd') {
            int i = va_arg(args, int);
            printf("%d\n", i);
        } else if (*fmt == 'c') {
            // note automatic conversion to integral type
            int c = va_arg(args, int);
            printf("%c\n", c);
        } else if (*fmt == 'f') {
            double d = va_arg(args, double);
            printf("%f\n", d);
        }
        ++fmt;
    }
 
    va_end(args);
}

int
main (void)
{
  printf("%d\n", add_em_up(3, 5, 5, 6)); 
  assert(add_em_up(3, 5, 5, 6) == 16);

  printf("%d\n", add_em_up(10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)); 
  assert(add_em_up(10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == 55);

  printf("%f\n", stddev(4, 25.0, 27.3, 26.9, 25.7)); 

  simple_printf("dcff", 3, 'a', 1.999, 42.5); 
  return 0;
}
