#include <utility>

int classify(int x)
{
  switch (x)
  {
  case 0:
    return 1;
  case 1:
    return 2;
  }
  std::unreachable();
}

int main()
{
  int x = 0;
  return classify(x) == 1 ? 0 : 1;
}
