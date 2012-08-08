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
   message("attempted to divide by zero"){assert(0);}

  const char *whato() const {return message;}

private:
  const char *message;
};

double quotient( int numerator, int denominator )
{
  if ( denominator == 0 )
    throw DivideByZeroException(); // terminate function

  return static_cast< double >( numerator ) / denominator;
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
    } // end try
    // exception handler handles a divide-by-zero exception
    catch ( DivideByZeroException &divideByZeroException ) {
      cout << "Exception occurred: " <<
          divideByZeroException.whato() << endl;
    } // end catch
    cout << "\nEnter two integers (end-of-file to end): ";
  }  // end while

  cout << endl;
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
