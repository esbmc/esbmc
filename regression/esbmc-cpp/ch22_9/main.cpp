// Fig. 22.5: fig22_05.cpp
// Demonstrating operator keywords.
#include <iostream>

using std::cout;
using std::endl;
using std::boolalpha;

#include <ciso646> 

int main()
{
   int a = 2;
   int b = 3;

   cout << boolalpha 
       <<   "   a and b: " << ( a and b )
       << "\n    a or b: " << ( a or b )
       << "\n     not a: " << ( not a )
       << "\na not_eq b: " << ( a not_eq b )
       << "\na bitand b: " << ( a bitand b )
       << "\na bit_or b: " << ( a bitor b )
       << "\n   a xor b: " << ( a xor b )
       << "\n   compl a: " << ( compl a )
       << "\na and_eq b: " << ( a and_eq b )
       << "\n a or_eq b: " << ( a or_eq b )
       << "\na xor_eq b: " << ( a xor_eq b ) << endl;
		
   return 0;

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
