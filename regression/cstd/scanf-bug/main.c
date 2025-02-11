#include <stdio.h>
int main(int argc, char *argv[])
{
  char reg_name[12];
  printf("Enter your username:");
  scanf("%s", reg_name); //Buffer overflow here
  printf("The program is now registered to %s.\n", reg_name);
  return 0;
}
