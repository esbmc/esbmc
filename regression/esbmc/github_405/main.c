struct a
{
  short symbol;
};
struct d
{
  struct a f;
};
int main()
{
  struct d *g;
  g->f;
  return 0;
}
