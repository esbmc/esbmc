//#include <assert.h>
//#include <stdio.h>

int main()
{
  printf("%s\n", __func__);
  printf("%s\n", __FUNCTION__);
//  printf("%s\n", __FUNCDNAME__);
//  printf("%s\n", L__FUNCTION__);
  printf("%s\n", __PRETTY_FUNCTION__);
//  printf("%s\n", __PRETTYFUNCTIONNOVIRTUAL__);
//  printf("%s\n", __FUNCSIG__);

  return 0;
}
