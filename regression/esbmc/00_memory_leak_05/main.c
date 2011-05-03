#include <stdlib.h>
#include <stdio.h>

int main(void) {
  char *chptr;
  char *chptr1;
  int i = 1;
  chptr = (char *) malloc(12);
  chptr1 = (char *) malloc (12);
  for ( i; i <= 13; i++ ) {
    chptr[i] = '?';        /* error when i = 13 invalid write */
    chptr1[i] = chptr[i];  /* error when i = 13 invalid read and write */
  }

  free(chptr1);
  free(chptr);
}
