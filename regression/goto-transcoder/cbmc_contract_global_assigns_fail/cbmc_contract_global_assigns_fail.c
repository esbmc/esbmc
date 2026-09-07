int g;

void setg(int v)
  __CPROVER_requires(v >= 0 && v < 100)
  __CPROVER_assigns(g)
  __CPROVER_ensures(g == v)
{
  g = v + 1;
}

int main() { return 0; }
