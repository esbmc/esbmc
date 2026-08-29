/* The decayed pointer designates the first element of a four-element array, so
   reading index 4 is out of bounds on either arm. */
char a[4] = {1, 2, 3, 4};
char b[4] = {5, 6, 7, 8};

int main(int argc, char **argv)
{
  char *c = argc == 1 ? a : b;
  return c[4];
}
