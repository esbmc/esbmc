// Fig. 8.9: fig08_09.cpp
// String class test program.
#include <iostream>

using std::cout;
using std::endl;

#include "string1.h"

int main()
{
   String s1( "happy" );
   String s2( " birthday" );
   String s3;

   // test overloaded equality and relational operators
   cout << "s1 is \"" << s1 << "\"; s2 is \"" << s2
        << "\"; s3 is \"" << s3 << '\"' 
        << "\n\nThe results of comparing s2 and s1:"
        << "\ns2 == s1 yields " 
        << ( s2 == s1 ? "true" : "false" )
        << "\ns2 != s1 yields " 
        << ( s2 != s1 ? "true" : "false" )
        << "\ns2 >  s1 yields " 
        << ( s2 > s1 ? "true" : "false" ) 
        << "\ns2 <  s1 yields " 
        << ( s2 < s1 ? "true" : "false" ) 
        << "\ns2 >= s1 yields "
        << ( s2 >= s1 ? "true" : "false" )
        << "\ns2 <= s1 yields " 
        << ( s2 <= s1 ? "true" : "false" );

   // test overloaded String empty (!) operator
   cout << "\n\nTesting !s3:\n";

   if ( !s3 ) {
      cout << "s3 is empty; assigning s1 to s3;\n";
      s3 = s1;  // test overloaded assignment
      cout << "s3 is \"" << s3 << "\"";
   }

   // test overloaded String concatenation operator
   cout << "\n\ns1 += s2 yields s1 = ";
   s1 += s2;  // test overloaded concatenation
   cout << s1;

   // test conversion constructor
   cout << "\n\ns1 += \" to you\" yields\n";
   s1 += " to you";  // test conversion constructor
   cout << "s1 = " << s1 << "\n\n";

   // test overloaded function call operator () for substring
   cout << "The substring of s1 starting at\n"
        << "location 0 for 14 characters, s1(0, 14), is:\n"
        << s1( 0, 14 ) << "\n\n";

   // test substring "to-end-of-String" option
   cout << "The substring of s1 starting at\n"
        << "location 15, s1(15, 0), is: "
        << s1( 15, 0 ) << "\n\n";  // 0 is "to end of string"

   // test copy constructor
   String *s4Ptr = new String( s1 );  
   cout << "\n*s4Ptr = " << *s4Ptr << "\n\n";

   // test assignment (=) operator with self-assignment
   cout << "assigning *s4Ptr to *s4Ptr\n";
   *s4Ptr = *s4Ptr;  // test overloaded assignment
   cout << "*s4Ptr = " << *s4Ptr << '\n';

   // test destructor
   delete s4Ptr;     

   // test using subscript operator to create lvalue
   s1[ 0 ] = 'H';      
   s1[ 6 ] = 'B';
   cout << "\ns1 after s1[0] = 'H' and s1[6] = 'B' is: "
        << s1 << "\n\n";

   // test subscript out of range
   cout << "Attempt to assign 'd' to s1[30] yields:" << endl;
   s1[ 30 ] = 'd';     // ERROR: subscript out of range

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