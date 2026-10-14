// #7907: a vector needs its own alignment, not just its lanes'.
typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  int buf[8] = {0};
  v4i c = *(v4i *)(buf + 1);
  return c[0];
}
