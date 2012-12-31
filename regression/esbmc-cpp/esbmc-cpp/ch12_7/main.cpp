// Fig. 12.9: fig12_09.cpp 
// Controlling precision of floating-point values.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;
using std::fixed;

#include <iomanip>

using std::setprecision;

#include <cmath>  // sqrt prototype

int main()
{
   double root2 = sqrt( 2.0 ); // calculate square root of 2
   int places;

   cout << "Square root of 2 with precisions 0-9." << endl
        << "Precision set by ios_base member-function "
        << "precision:" << endl;

   cout << fixed; // use fixed precision

   // display square root using ios_base function precision
   for ( places = 0; places <= 9; places++ ) {
      cout.precision( places );
      cout << root2 << endl;
   }

   cout << "\nPrecision set by stream-manipulator " 
        << "setprecision:" << endl;

   // set precision for each digit, then display square root
   for ( places = 0; places <= 9; places++ )
      cout << setprecision( places ) << root2 << endl;

   return 0;

} // end main

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