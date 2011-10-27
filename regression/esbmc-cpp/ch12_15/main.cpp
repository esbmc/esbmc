// Fig. 12.18: fig12_18.cpp 
// Displaying floating-point values in system default,
// scientific and fixed formats.
#include <iostream>

using std::cout;
using std::endl;
using std::scientific;
using std::fixed;

int main()
{
   double x = 0.001234567;
   double y = 1.946e9;

   // display x and y in default format
   cout << "Displayed in default format:" << endl
        << x << '\t' << y << endl;

   // display x and y in scientific format
   cout << "\nDisplayed in scientific format:" << endl
        << scientific << x << '\t' << y << endl;

   // display x and y in fixed format
   cout << "\nDisplayed in fixed format:" << endl
        << fixed << x << '\t' << y << endl;

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