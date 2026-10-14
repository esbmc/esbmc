// strsep had no declaration or model, so a C call was implicitly declared as
// returning int and verified against an unconstrained result (github #7548).
#include <assert.h>
#include <string.h>

int main(void)
{
  char buffer[] = "x;y";
  char *rest = buffer;
  char *token = strsep(&rest, ";");
  assert(token == buffer && buffer[1] == '\0' && rest == buffer + 2);
  token = strsep(&rest, ";");
  assert(token == buffer + 2 && rest == NULL);
  return 0;
}
