template <typename T>
bool check()
{
  return requires(T t) { t.value; };
}

struct S
{
};

int main()
{
  __ESBMC_assert(!check<S>(), "");
}
