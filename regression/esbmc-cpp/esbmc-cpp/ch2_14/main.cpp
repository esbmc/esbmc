// Fig. 2.21: fig02_21.cpp
// Calculating compound interest.
#include <iostream>

using std::cout;
using std::endl;
using std::ios;
using std::fixed;

#include <iomanip>

using std::setw;
using std::setprecision;

#include <cmath>  // enables program to use function pow

// function main begins program execution
int main()
{
   double amount;              // amount on deposit
   double principal = 1000.0;  // starting principal
   double rate = .05;          // interest rate

   // output table column heads
   cout << "Year" << setw( 21 ) << "Amount on deposit" << endl;

   // set floating-point number format
   cout << fixed << setprecision( 2 );

   // calculate amount on deposit for each of ten years
   for ( int year = 1; year <= 10; year++ ) {

      // calculate new amount for specified year
      amount = principal * pow( 1.0 + rate, year );

      // output one table row
      cout << setw( 4 ) << year 
           << setw( 21 ) << amount << endl;

   } // end for 

   return 0;   // indicate successful termination

} // end function main


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
