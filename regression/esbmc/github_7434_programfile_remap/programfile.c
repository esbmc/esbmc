/* Regression for #7434: --witness-programfile remaps a line by matching the
 * text of the verified file's line into this file, so it is meaningful only
 * for a step that belongs to the verified file. The violation is on line 6 of
 * helper.h; before the fix ESBMC looked up main.c's line 6 instead, found
 * "return helper_div(10, d);" at line 14 here, and reported startline 14. */
/* pad */
/* pad */
/* pad */
#include "helper.h"

int main(void)
{
  int d = 0;
  return helper_div(10, d);
}
