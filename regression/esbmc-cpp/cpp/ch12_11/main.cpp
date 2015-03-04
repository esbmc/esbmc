// Fig. 12.14: fig12_14.cpp 
// Demonstrating left justification and right justification.
#include <iostream>

using std::cout;
using std::endl;
using std::left;
using std::right;

#include <iomanip>

using std::setw;

int main()
{
   int x = 12345;

   // display x right justified (default)
   cout << "Default is right justified:" << endl
        << setw( 10 ) << x;

   // use left manipulator to display x left justified
   cout << "\n\nUse std::left to left justify x:\n"
        << left << setw( 10 ) << x;

   // use right manipulator to display x right justified
   cout << "\n\nUse std::right to right justify x:\n"
        << right << setw( 10 ) << x << endl;

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