// Fig. 18.32: fig18_32.cpp
// Using strspn.
#include <iostream>

using std::cout;
using std::endl;

#include <cstring>  // strspn prototype

int main()
{
   const char *string1 = "The value is 3.14159";
   const char *string2 = "aehils Tuv";

   cout << "string1 = " << string1 
        << "\nstring2 = " << string2
        << "\n\nThe length of the initial segment of string1\n"
        << "containing only characters from string2 = "
        << strspn( string1, string2 ) << endl;

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