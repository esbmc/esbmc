/* Regression for #7434: a witness step must name the file it belongs to.
 * The division by zero is on line 6 of helper.h, not line 6 of main.c, so a
 * GraphML edge carrying a bare startline sends a consumer to the wrong file. */
#include "helper.h"

int main(void)
{
  int d = 0;
  return helper_div(10, d);
}
