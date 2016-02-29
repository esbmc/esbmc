#include <string.h>
#include <locale.h>
#include <assert.h>
 
int main(void)
{
  setlocale(LC_COLLATE, "cs_CZ.iso88592");
 
  const char* s1 = "hrnec";
  const char* s2 = "chrt";
 
  assert(strcoll(s1, s2) < 0);

  return 0;
}
