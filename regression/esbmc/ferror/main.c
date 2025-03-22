#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
 
int main(void)
{
    char* fname;
    FILE* f;

    if (fname != NULL)
      f = fopen(fname, "wb");
    
    if (f == NULL)
      return -1;
    else
    {
      fputs("\xff\xff\n", f); // not a valid UTF-8 character sequence 
      fclose(f);
    }

    if (fname != NULL) 
      f = fopen(fname, "rb");

    if (f == NULL)
      return -1;
    else
    { 
      if (feof(f))
        assert(0 && "EOF indicator set");
      if (ferror(f))
        assert(0 && "Error indicator set");
    }
    return 0;
}
