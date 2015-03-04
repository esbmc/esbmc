// Fig. 12.20: fig12_20.cpp
// Demonstrating stream-manipulators boolalpha and noboolalpha.
#include <iostream>

using std::cout;
using std::endl;
using std::cin;
using std::boolalpha;
using std::noboolalpha;

int main()
{
   bool booleanValue = true;

   // display default true booleanValue
   cout << "booleanValue is " << booleanValue << endl;

   // display booleanValue after using boolalpha
   cout << "booleanValue (after using boolalpha) is "
        << boolalpha << booleanValue << endl << endl;

   cout << "switch booleanValue and use noboolalpha" << endl;
   booleanValue = false; // change booleanValue
   cout << noboolalpha << endl; // use noboolalpha

   // display default false booleanValue after using noboolalpha
   cout << "booleanValue is " << booleanValue << endl;

   // display booleanValue after using boolalpha again
   cout << "booleanValue (after using boolalpha) is "
        << boolalpha << booleanValue << endl;

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