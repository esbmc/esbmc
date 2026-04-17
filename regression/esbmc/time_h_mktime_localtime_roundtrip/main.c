#include <time.h>
#include <assert.h>

int main()
{
  time_t original = 1000000;
  struct tm *broken = localtime(&original);
  if (broken != (void *)0)
  {
    time_t reconstructed = mktime(broken);
    assert(difftime(original, reconstructed) == 0.0);
  }
  return 0;
}
