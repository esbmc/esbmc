#include <string.h>
#include <locale.h>
 
int main(void)
{
  setlocale(LC_COLLATE, "cs_CZ.iso88592");
 
  const char *in1 = "hrnec";
  char out1[1+strxfrm(NULL, in1, 0)];
  strxfrm(out1, in1, sizeof out1);
 
  const char *in2 = "chrt";
  char out2[1+strxfrm(NULL, in2, 0)];
  strxfrm(out2, in2, sizeof out2);
 
  printf("In the Czech locale: ");
  assert(strcmp(out1, out2) < 0);

  return 0;
}
