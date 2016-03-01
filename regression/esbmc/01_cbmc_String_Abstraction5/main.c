#include <string.h>

int main()
{
  char a[100];
  const char *p;
  unsigned int i;

  // this should work
  p=strerror(1);

  // this should work
  i=strlen(p);

  // this sould work
  if(i<sizeof(a))
    strcpy(a, p);
    
  // and even this should work
  char ch;
  ch=*p;
}
