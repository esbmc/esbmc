#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
  if (argc<2) {
    fprintf(stderr, "Program name is: %s\n", argv[0]);
  }
}
