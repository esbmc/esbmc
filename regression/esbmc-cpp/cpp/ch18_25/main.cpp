// Fig. 18.37: fig18_37.cpp
// Using memcmp.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

#include <cstring>  // memcmp prototype

int main()
{
   char s1[] = "ABCDEFG";
   char s2[] = "ABCDXYZ";

   cout << "s1 = " << s1 << "\ns2 = " << s2 << endl
        << "\nmemcmp(s1, s2, 4) = " << setw( 3 ) 
        << memcmp( s1, s2, 4 ) << "\nmemcmp(s1, s2, 7) = " 
        << setw( 3 ) << memcmp( s1, s2, 7 )
        << "\nmemcmp(s2, s1, 7) = " << setw( 3 ) 
        << memcmp( s2, s1, 7 ) << endl;

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