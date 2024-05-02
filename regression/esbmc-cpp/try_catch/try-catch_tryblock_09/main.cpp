#include<cassert>

int main()
{
try {
  throw 5;
  return 0;
}
catch (int) {
  return -1;
}
}
