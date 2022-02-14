#include <cheri/cheric.h>

int main(int argc, char *argv[])
{
  char array[2];
  char *__capability arrayp;
  int ii;
  arrayp = cheri_ptr(array, sizeof(array));
  for(ii = 0; ii < sizeof(array) + 1; ii++)
  {
    arrayp[ii] = 0;
  }
  return (0);
}
