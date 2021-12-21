#include <stdio.h>
#include <string.h>

int main()
{
  // this should work
  FILE *f;
  char buffer[10];
  unsigned int i;
  
  f=fopen("asd", "xzy");

  if(fgets(buffer, 10, f)!=0)
  {
    i=strlen(buffer);
  }

  fclose(f);
}
