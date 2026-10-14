/* Regression for #7434: a step that belongs to the verified file has its line
 * remapped into the program file, so naming the verified file alongside that
 * remapped line would point a consumer past the end of it. This file is 8
 * lines long; the violation remaps to line 12 of programfile.c. */
int main(void)
{
  int d = 0;
  return 10 / d;
}
