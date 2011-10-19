// Fig. 18.22: fig18_22.cpp
// Using atoi.
#include <iostream>

using std::cout;
using std::endl;

#include <cstdlib>  // atoi prototype

int main()
{
   int i = atoi( "2593" );

   cout << "The string \"2593\" converted to int is " << i
        << "\nThe converted value minus 593 is " << i - 593 
        << endl;

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