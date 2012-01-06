// Fig. 8.13: fig08_13.cpp
// Standard library string class test program.
#include <iostream>

using std::cout;
using std::endl;

#include <string>

using std::string;

int main()
{
   string s1( "happy" );
   string s2( " birthday" );
   string s3("happy birthday");


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
		  << s2.length()
		  << s1+s2
        << ( s2 <= s1 ? "true" : "false" );

   // test string member function empty 
   cout << "\n\nTesting s3.empty():\n";

   if ( s3.empty() ) {
      cout << "s3 is empty; assigning s1 to s3;\n";
      s3 = s1;  // assign s1 to s3
      cout << "s3 is \"" << s3 << "\"";
   }

   // test overloaded string concatenation operator
   cout << "\n\ns1 += s2 yields s1 = ";
   s1 += s2;  // test overloaded concatenation
   cout << s1;

   // test overloaded string concatenation operator 
   // with C-style string
   cout << "\n\ns1 += \" to you\" yields\n";
   s1 += " to you"; 
   cout << "s1 = " << s1 << "\n\n";

   // test string member function substr
   cout << "The substring of s1 starting at location 0 for\n"
        << "14 characters, s1.substr(0, 14), is:\n"
        << s1.substr( 0, 14 ) << "\n\n";

   // test substr "to-end-of-string" option
   cout << "The substring of s1 starting at\n"
        << "location 15, s1.substr(15), is:\n"
        << s1.substr( 15 ) << '\n'; 

   // test copy constructor
   string *s4Ptr = new string( s1 );  
   cout << "\n*s4Ptr = " << *s4Ptr << "\n\n";

   // test assignment (=) operator with self-assignment
   cout << "assigning *s4Ptr to *s4Ptr\n";
   *s4Ptr = *s4Ptr;  
   cout << "*s4Ptr = " << *s4Ptr << '\n';

   // test destructor
   delete s4Ptr;     

   // test using subscript operator to create lvalue
   s1[ 0 ] = 'H';      
   s1[ 6 ] = 'B';
   cout << "\ns1 after s1[0] = 'H' and s1[6] = 'B' is: "
        << s1 << "\n\n";

   // test subscript out of range with string member function "at"
   cout << "Attempt to assign 'd' to s1.at( 30 ) yields:" << endl;
   s1.at( 30 ) = 'd';     // ERROR: subscript out of range

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
