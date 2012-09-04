// Fig. 5.11: fig05_11.cpp
// Printing a string one character at a time using
// a non-constant pointer to constant data.
#include <iostream>

using std::cout;
using std::endl;

void printCharacters( const char * );

int main()
{
   char phrase[] = "print characters of a string";

   cout << "The string is:\n";
   printCharacters( phrase );
   cout << endl;

   return 0;  // indicates successful termination

} // end main

// sPtr cannot modify the character to which it points, 
// i.e., sPtr is a "read-only" pointer
void printCharacters( const char *sPtr )
{
   for ( ; *sPtr != '\0'; sPtr++ )   // no initialization
      cout << *sPtr;

} // end function printCharacters

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
