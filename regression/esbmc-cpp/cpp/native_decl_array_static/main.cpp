#include <cassert>

int dtor_runs = 0;

struct Guard
{
  ~Guard()
  {
    ++dtor_runs;
  }
};

static int counter = 7;

// The three declaration shapes this handler delegates to convert_decl rather
// than reproducing natively: a static-lifetime symbol, an array (conservatively
// excluded because it may be a VLA), and a destructible type. Each must still
// declare, initialise and destroy exactly as the legacy body path does, while
// the statements around them convert natively.
int sum_array(int n)
{
  Guard g;
  int values[4] = {1, 2, 3, 4};
  int total = 0;
  for (int i = 0; i < 4; ++i)
    total += values[i];
  return total + n;
}

int main()
{
  assert(counter == 7);
  counter = 9;
  assert(counter == 9);

  assert(sum_array(0) == 10);
  assert(dtor_runs == 1);

  assert(sum_array(5) == 15);
  assert(dtor_runs == 2);
  return 0;
}
