#include <stdio.h>
int main(int argc, char *argv[])
{
  char reg_name[12];
  printf("Enter your username:");
  scanf("%11s", reg_name); 
  printf("The program is now registered to %s.\n", reg_name);
  return 0;
}
