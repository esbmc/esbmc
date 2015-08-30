// Fig. 5.30: fig05_30.cpp
// Using strcmp and strncmp.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

#include <cstring>  // prototypes for strcmp and strncmp

int main()
{
   char *s1 = "Happy New Year";
   char *s2 = "Happy New Year";
   char *s3 = "Happy Holidays";
 
   cout << "s1 = " << s1 << "\ns2 = " << s2
        << "\ns3 = " << s3 << "\n\nstrcmp(s1, s2) = " 
        << setw( 2 ) << strcmp( s1, s2 ) 
        << "\nstrcmp(s1, s3) = " << setw( 2 ) 
        << strcmp( s1, s3 ) << "\nstrcmp(s3, s1) = " 
        << setw( 2 ) << strcmp( s3, s1 );

   cout << "\n\nstrncmp(s1, s3, 6) = " << setw( 2 ) 
        << strncmp( s1, s3, 6 ) << "\nstrncmp(s1, s3, 7) = " 
        << setw( 2 ) << strncmp( s1, s3, 7 ) 
        << "\nstrncmp(s3, s1, 7) = " 
        << setw( 2 ) << strncmp( s3, s1, 7 ) << endl;

   return 0;  // indicates successful termination

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
