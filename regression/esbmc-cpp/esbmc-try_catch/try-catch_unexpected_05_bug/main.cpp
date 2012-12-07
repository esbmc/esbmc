#include<exception>
#include<cassert>
using namespace std;

void myunexpected()
{
}

int main() throw(char)
{
  set_unexpected(myunexpected);
  throw 5;
  return 0;
}
