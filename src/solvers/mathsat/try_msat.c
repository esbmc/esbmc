#include <mathsat.h>

int main()
{
  const char *msat_version = msat_get_version();
  printf("%s", msat_version);
  msat_free(msat_version);
  return 0;
}