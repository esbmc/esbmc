// Fig. 18.36: fig18_36.cpp
// Using memmove.
#include <iostream>

using std::cout;
using std::endl;

#include <cstring>  // memmove prototype

int main()
{
   char x[] = "Home Sweet Home";

   cout << "The string in array x before memmove is: " << x;
   cout << "\nThe string in array x after memmove is:  "
        << static_cast< char * >( memmove( x, &x[ 5 ], 10 ) ) 
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