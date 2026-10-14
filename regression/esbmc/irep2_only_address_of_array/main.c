/* `&Q` on an array is `&Q[0]`: the pointer designates the first element, not
   the array object. */
int Q[3];
int *b = (int *)&Q;

int main(void)
{
  Q[0] = 7;
  return b[0];
}
