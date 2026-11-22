/* A struct with no members has no field to split into, so
 * --scalarize-structs (#7521) leaves it alone and splits only the other. */
struct empty
{
};

struct pair
{
  int x;
  int y;
};

int main(void)
{
  struct empty a, b;
  a = b;
  struct pair p = {1, 2};
  __ESBMC_assert(p.x + p.y == 3, "sum");
  return 0;
}
