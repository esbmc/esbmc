// Fig. 18.21: fig18_21.cpp
// Using atof.
#include <iostream>

using std::cout;
using std::endl;

#include <cstdlib>  // atof prototype

int main()
{
   double d = atof( "99.0" );

   cout << "The string \"99.0\" converted to double is "
        << d << "\nThe converted value divided by 2 is "
        << d / 2.0 << endl;

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