// Fig. 21.38: fig21_38.cpp
// Standard library algorithms min and max.
#include <iostream>

using std::cout;
using std::endl;

#include <algorithm>

int main()
{
   cout << "The minimum of 12 and 7 is: " 
        << std::min( 12, 7 );
   cout << "\nThe maximum of 12 and 7 is: " 
        << std::max( 12, 7 );
   cout << "\nThe minimum of 'G' and 'Z' is: " 
        << std::min( 'G', 'Z' );
   cout << "\nThe maximum of 'G' and 'Z' is: " 
        << std::max( 'G', 'Z' ) << endl; 

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