// Fig. 12.13: fig12_13.cpp 
// Using showpoint to control the printing of 
// trailing zeros and decimal points for doubles.
#include <iostream>

using std::cout;
using std::endl;
using std::showpoint;

int main()
{
   // display double values with default stream format
   cout << "Before using showpoint" << endl
        << "9.9900 prints as: " << 9.9900 << endl
        << "9.9000 prints as: " << 9.9000 << endl
        << "9.0000 prints as: " << 9.0000 << endl << endl;

   // display double value after showpoint
   cout << showpoint
        << "After using showpoint" << endl
        << "9.9900 prints as: " << 9.9900 << endl
        << "9.9000 prints as: " << 9.9000 << endl
        << "9.0000 prints as: " << 9.0000 << endl;

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