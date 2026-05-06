#include <assert.h>

// Regression for the GraphML/YAML witness-fanout bug flagged on PR
// #4310: violation_graphml_goto_trace and violation_yaml_goto_trace
// previously read the output path from `options` directly, so calling
// them N times during --all-witnesses overwrote the same file. The fix
// adds an output_path_override parameter and the bmc loop fans out
// per-witness filenames (`<ce_index>-<base>`).

int main(void)
{
  int x;
  if (x > 0)
    x--;
  else
    x++;
  assert(x != 0);
  return 0;
}
