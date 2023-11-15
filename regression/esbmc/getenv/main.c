#include <stdio.h>
#include <stdlib.h>

int main()
{
  // Test Case 1: Valid environment variable
  char *path = getenv("PATH");
  if(path != NULL)
    printf("Test Case 1: PATH = %s\n", path);
  else
    printf("Test Case 1: PATH environment variable not found\n");

  // Test Case 2: Non-existent environment variable
  char *nonExistent = getenv("NON_EXISTENT_VARIABLE");
  if(nonExistent != NULL)
    printf("Test Case 2: Found non-existent variable: %s\n", nonExistent);
  else
    printf("Test Case 2: Non-existent variable not found\n");

  // Test Case 3: Reading HOME environment variable
  char *home = getenv("HOME");
  if(home != NULL)
    printf("Test Case 3: HOME = %s\n", home);
  else
    printf("Test Case 3: HOME environment variable not found\n");

  return 0;
}
