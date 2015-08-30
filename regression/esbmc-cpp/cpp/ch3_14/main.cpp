// Fig. 3.15: fig03_15.cpp
// Recursive fibonacci function.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

unsigned long fibonacci( unsigned long ); // function prototype

int main()
{
   unsigned long result, number;

   // obtain integer from user
   cout << "Enter an integer: ";
   cin >> number;

   // calculate fibonacci value for number input by user
   result = fibonacci( number );

   // display result
   cout << "Fibonacci(" << number << ") = " << result << endl;

   return 0;  // indicates successful termination

} // end main

// recursive definition of function fibonacci
unsigned long fibonacci( unsigned long n )
{
   // base case
   if ( n == 0 || n == 1 )  
      return n;

   // recursive step
   else             
      return fibonacci( n - 1 ) + fibonacci( n - 2 );

} // end function fibonacci


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
