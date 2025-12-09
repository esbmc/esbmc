#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
  char name[20] = "";

  if (argc > 1 && argv[1] != NULL)
  {
    memcpy(name, argv[1], 30);
  }
}
