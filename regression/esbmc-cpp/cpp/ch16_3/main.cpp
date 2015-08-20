// Fig. 16.9: querystring.cpp
// Demonstrating QUERY_STRING.
#include <iostream>

using std::cout;

#include <string>

using std::string;

#include <cstdlib>

int main()
{
   string query = getenv( "QUERY_STRING" );
   
   // output header
   cout << "Content-type: text/html\n\n";
   
   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>"
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\" \"http://www.w3.org/TR/xhtml1"
        << "/DTD/xhtml1-transitional.dtd\">";
   
   // output html element and some of its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Name/Value Pairs</title></head>"
        << "<body>";
   
   cout << "<h2>Name/Value Pairs</h2>";
   
   // if query contained no data
   if ( query == "" )
      cout << "Please add some name-value pairs to the URL "
           << "above.<br/>Or try "
           << "<a href=\"querystring.cgi?name=Joe&age=29\">"
           << "this</a>.";
   
   // user entered query string
   else
      cout << "<p>The query string is: " << query << "</p>";
   
   cout << "</body></html>";
   
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
