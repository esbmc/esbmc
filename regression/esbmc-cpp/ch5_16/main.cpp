// Fig. 5.21: fig05_21.cpp
// Copying a string using array notation
// and pointer notation.
#include <iostream>

using std::cout;
using std::endl;

void copy1( char *, const char * );  // prototype
void copy2( char *, const char * );  // prototype

int main()
{
   char string1[ 10 ];
   char *string2 = "Hello";
   char string3[ 10 ];
   char string4[] = "Good Bye";

   copy1( string1, string2 );
   cout << "string1 = " << string1 << endl;

   copy2( string3, string4 );
   cout << "string3 = " << string3 << endl;

   return 0;  // indicates successful termination

} // end main

// copy s2 to s1 using array notation
void copy1( char *s1, const char *s2 )
{
   for ( int i = 0; ( s1[ i ] = s2[ i ] ) != '\0'; i++ )
      ;   // do nothing in body

} // end function copy1

// copy s2 to s1 using pointer notation
void copy2( char *s1, const char *s2 )
{
   for ( ; ( *s1 = *s2 ) != '\0'; s1++, s2++ )
      ;   // do nothing in body

} // end function copy2

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
