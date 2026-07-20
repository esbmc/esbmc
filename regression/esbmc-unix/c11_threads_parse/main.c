/* Regression for https://github.com/esbmc/esbmc/issues/3449 :
 * ESBMC must be able to parse code that includes <threads.h>. */
#include <threads.h>

int main(void)
{
  return 0;
}
