// Fig. 5.6: fig05_06.cpp
// Compound interest calculations with for.
#include <iostream>
#include <iomanip>
#include <cmath> // standard math library
using namespace std;

int main()
{
   double amount; // amount on deposit at end of each year
   double principal = 1000.0; // initial amount before interest
   double rate = .05; // annual interest rate

   // display headers
   cout << "Year" << setw( 21 ) << "Amount on deposit" << endl;

   // set floating-point number format
   cout << fixed << setprecision( 2 );

   // calculate amount on deposit for each of ten years
   for ( unsigned int year = 1; year <= 10; ++year ) 
   {
      // calculate new amount for specified year
      amount = principal * pow( 1.0 + rate, year );

      // display the year and the amount
      cout << setw( 4 ) << year << setw( 21 ) << amount << endl;
   } // end for 
} // end main


/**************************************************************************
 * (C) Copyright 1992-2014 by Deitel & Associates, Inc. and               *
 * Pearson Education, Inc. All Rights Reserved.                           *
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
 **************************************************************************/
