#include <string.h>
const int t[2] = {1, 2};
int main()
{
  int s[2] = {7, 8};
  memcpy((void *)t, s, sizeof t);
  return 0;
}
