// Fig. 16.2: fig16_02.cpp
// Demonstrating string comparison capabilities.
#include <iostream>

using std::cout;
using std::endl;

#include <string>
#include <cassert>

using std::string;

int main()
{
   string string1( "Testing the comparison functions." );
   string string2( "Hello" ); 
   string string3( "stinger" );
   string string4( string2 );
   
   cout << "string1: " << string1 << "\nstring2: " << string2
        << "\nstring3: " << string3 << "\nstring4: " << string4
        << "\n\n";


   // comparing string1 and string4
   if ( string1 == string4 )
      cout << "string1 == string4\n";
   else { // string1 != string4 
      if ( string1 > string4 )
         cout << "string1 > string4\n";
      else // string1 < string4
         cout << "string1 < string4\n";
   }

   // comparing string1 and string2
   int result = string1.compare( string2 );

   if ( result == 0 )
      cout << "string1.compare( string2 ) == 0\n";
   else  // result != 0
      if ( result > 0 )
         cout << "string1.compare( string2 ) > 0\n";
      else  // result < 0
         cout << "string1.compare( string2 ) < 0\n";

   // comparing string1 (elements 2-5) and string3 (elements 0-5)
   result = string1.compare( 2, 5, string3, 0, 5 );

   if ( result == 0 )
      cout << "string1.compare( 2, 5, string3, 0, 5 ) == 0\n";
   else  // result != 0
      if ( result > 0 )
         cout << "string1.compare( 2, 5, string3, 0, 5 ) > 0\n";
      else  // result < 0
         cout << "string1.compare( 2, 5, string3, 0, 5 ) < 0\n";

   // comparing string2 and string4
   result = string4.compare( 0, string2.length(), string2 );

   if ( result == 0 )
      cout << "string4.compare( 0, string2.length(), " 
           << "string2 ) == 0" << endl;
   else  // result != 0
      if ( result > 0 )
         cout << "string4.compare( 0, string2.length(), "
              << "string2 ) > 0" << endl;
      else  // result < 0
         cout << "string4.compare( 0, string2.length(), "
              << "string2 ) < 0" << endl;
   
   // comparing string2 and string4
   result = string2.compare( 0, 3, string4 );

   if ( result == 0 )
      cout << "string2.compare( 0, 3, string4 ) == 0" << endl;
   else  // result != 0
      if ( result > 0 )
         cout << "string2.compare( 0, 3, string4 ) > 0" << endl;
      else  // result < 0
         cout << "string2.compare( 0, 3, string4 ) < 0" << endl;

   return 0;

}  // end main

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
