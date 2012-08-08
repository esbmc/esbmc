#include <exception>
using std::exception;

// DivideByZeroException objects should be thrown by functions
// upon detecting division-by-zero exceptions
class DivideByZeroException : public exception {
public:
  DivideByZeroException() :
    message("attempted to divide by zero"){}

  const char *whato() const {return message;}

private:
  const char *message;
};  // end class DivideByZeroException

// perform division and throw DivideByZeroException object if
// divide-by-zero exception occurs
double quotient( int numerator, int denominator )
{
  // throw DivideByZeroException if trying to divide by zero
  if ( denominator == 0 )
    throw DivideByZeroException(); // terminate function

  // return division result
  return static_cast< double >( numerator ) / denominator;

}  // end function quotient

int main()
{
  int number1;    // user-specified numerator
  int number2;    // user-specified denominator
  int result;  // result of division

  try {
    result = quotient( number1, number2 );
  } // end try
  // exception handler handles a divide-by-zero exception
  catch ( DivideByZeroException &divideByZeroException ) { } // end catch

  return 0;  // terminate normally
}  // end main


/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/
