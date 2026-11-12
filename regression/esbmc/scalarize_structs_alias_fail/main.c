/* A struct whose address is taken keeps its struct under --scalarize-structs
 * (#7521): a write through the pointer must reach the field that is read. */
struct pair
{
  int x;
  int y;
};

int main(void)
{
  struct pair p = {1, 2};
  struct pair *q = &p;
  q->y = 7;
  __ESBMC_assert(p.y != 7, "y written through q");
  return 0;
}
