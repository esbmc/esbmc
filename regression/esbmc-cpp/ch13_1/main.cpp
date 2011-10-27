// Fig. 13.1: fig13_01.cpp
// A simple exception-handling example that checks for
// divide-by-zero exceptions.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include <exception>

using std::exception;

// DivideByZeroException objects should be thrown by functions
// upon detecting division-by-zero exceptions
class DivideByZeroException : public exception {

public:

   // constructor specifies default error message
   DivideByZeroException::DivideByZeroException()
      : exception( "attempted to divide by zero" ) {}

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
   double result;  // result of division

   cout << "Enter two integers (end-of-file to end): ";

   // enable user to enter two integers to divide
   while ( cin >> number1 >> number2 ) {
   
      // try block contains code that might throw exception
      // and code that should not execute if an exception occurs
      try {
         result = quotient( number1, number2 );
         cout << "The quotient is: " << result << endl;

      } // end try

      // exception handler handles a divide-by-zero exception
      catch ( DivideByZeroException &divideByZeroException ) {
         cout << "Exception occurred: " << 
            divideByZeroException.what() << endl;

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