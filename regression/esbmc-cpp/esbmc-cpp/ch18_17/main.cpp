// Fig. 18.28: fig18_28.cpp
// Using strchr.
#include <iostream>

using std::cout;
using std::endl;

#include <cstring>  // strchr prototype

int main()
{
   const char *string1 = "This is a test";
   char character1 = 'a';
   char character2 = 'z';

   if ( strchr( string1, character1 ) != NULL )
      cout << '\'' << character1 << "' was found in \""
           << string1 << "\".\n";
   else
      cout << '\'' << character1 << "' was not found in \""
           << string1 << "\".\n";

   if ( strchr( string1, character2 ) != NULL )
      cout << '\'' << character2 << "' was found in \""
           << string1 << "\".\n";
   else
      cout << '\'' << character2 << "' was not found in \""
           << string1 << "\"." << endl;

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