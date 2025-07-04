#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char name[20]="";

  if (argc > 1 && argv[1] != NULL) {  
    strncpy(name, argv[1], 10);
  }

  printf("Hello, %s!\n", name);
  return 0;
}
