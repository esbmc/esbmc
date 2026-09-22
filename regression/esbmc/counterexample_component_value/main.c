// The holding counterpart of counterexample_component_value_fail: reporting a
// component's value is a trace concern and must not move a verdict.
struct sample
{
  int total;
  int reading[4];
};

struct sample s;

int main(void)
{
  for (int i = 0; i < 4; i++)
  {
    s.reading[i] = i * 3;
    s.total = s.total + s.reading[i];
  }

  __ESBMC_assert(s.total == 18, "total is not 18");
  return 0;
}
