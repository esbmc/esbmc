#include <iostream>
#include <cassert>
using std::cout;
using std::cin;
using std::endl;
#include <exception>
using std::exception;


class DivideByZeroException : public exception {
public:
  DivideByZeroException() :
    message("attempted to divide by zero"){/*assert(0);*/}

  const char *whato() const {return message;}

private:
  const char *message;
};

double quotient( int numerator, int denominator )
{
  if ( denominator == 0 )
    throw DivideByZeroException();

  return static_cast< double >( numerator );
}

int main()
{
  int number1;
  int number2;
  int result;

  cout << "Enter two integers (end-of-file to end): ";

  while ( cin >> number1 >> number2 ) {
    try {
      result = number1;
      result = quotient( number1, number2 );
      cout << "The quotient is: " << result << endl;
    }
    catch ( DivideByZeroException &divideByZeroException ) {
      cout << "Exception occurred: " <<
          divideByZeroException.whato() << endl;
      assert(0);
    }
    cout << "\nEnter two integers (end-of-file to end): ";
  }

  cout << endl;
  return 0;
}
