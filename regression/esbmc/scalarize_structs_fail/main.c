/* A field written through its own variable under --scalarize-structs (#7521)
 * shows in the counterexample under the field's name. */
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
  __ESBMC_assert(s.total != 18, "total is 18");
  return 0;
}
