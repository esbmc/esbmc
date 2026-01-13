// Test __ESBMC_old() with struct fields

struct Point {
  int x;
  int y;
};

struct Point p = {0, 0};

void move_point(int dx, int dy)
{
  __ESBMC_requires(dx >= 0 && dy >= 0);
  __ESBMC_ensures(p.x == __ESBMC_old(p.x) + dx);
  __ESBMC_ensures(p.y == __ESBMC_old(p.y) + dy);

  p.x += dx;
  p.y += dy;
}

int main()
{
  p.x = 10;
  p.y = 20;
  move_point(5, 3);
  assert(p.x == 15);
  assert(p.y == 23);
  return 0;
}
