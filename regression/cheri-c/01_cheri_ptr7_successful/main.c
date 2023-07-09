#include <assert.h>
#include <cheri/cheric.h>

char array[2];

int main(int argc, char *argv[])
{
  char *__capability arrayp;
  int ii;
  arrayp = cheri_ptr(array, sizeof(array));
  for(ii = 0; ii < sizeof(array); ii++)
  {
    assert(arrayp[ii] == 0);
  }
  return 0;
}
