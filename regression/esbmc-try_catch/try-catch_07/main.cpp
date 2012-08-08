#include <iostream>
using namespace std;

#include <exception>
using std::exception;

#include <cassert>

void throwException()
{
  int numerador = 2;
  int denominador = 0;

  try {
    if (denominador == 0)
      throw 1;
  }
  catch ( int ) {
    denominador = 1;
  }

  int result = numerador/denominador;
  cout << result << endl;
}

int main()
{
  throwException();
  return 0;
}
