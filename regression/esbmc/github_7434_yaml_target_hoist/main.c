/* Regression for #7434: README-YAML.md requires a waypoint's file_name to be
 * one of the task's input files. The violation is inside the memcpy
 * operational model, so the target waypoint must be hoisted to the innermost
 * call site that is an input file -- here main.c line 9 -- rather than
 * claiming main.c at the model's own line number. */
#include <string.h>

int main(void)
{
  char dst[4];
  char src[8] = "1234567";
  memcpy(dst, src, 8);
  return dst[0];
}
