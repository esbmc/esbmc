/* main itself lives in this header, so neither the violated assertion nor any
 * frame below it names an input file and the target waypoint cannot be
 * hoisted. README-YAML.md requires file_name to be one of task.input_files, so
 * the target is attributed to main.c rather than to body.h, which a validator
 * was never given. The line stays this file's -- a line the validator cannot
 * match only costs confirmation, whereas a foreign file_name is ill-formed. */
#include <assert.h>
int __VERIFIER_nondet_int(void);

int main(void)
{
  int v = __VERIFIER_nondet_int();
  assert(v != 7);
  return v;
}
