// Fig. 16.11: getquery.cpp
// Demonstrates GET method with XHTML form.
#include <iostream>

using std::cout;

#include <string>

using std::string;

#include <cstdlib>

int main()
{
   string nameString = "";
   string wordString = "";
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
        << "<head><title>Using GET with Forms</title></head>"
        << "<body>";
   
   // output xhtml form
   cout << "<p>Enter one of your favorite words here:</p>"
        << "<form method = \"get\" action = \"getquery.cgi\">"
        << "<input type = \"text\" name = \"word\"/>"
        << "<input type = \"submit\" value = \"Submit Word\"/>"
        << "</form>";
   
   // query is empty
   if ( query == "" )
      cout << "<p>Please enter a word.</p>";
   
   // user entered query string
   else {
      int wordLocation = query.find_first_of( "word=" ) + 5;

      wordString = query.substr( wordLocation );
      
      // no word was entered
      if ( wordString == "" )
         cout << "<p>Please enter a word.</p>";
      
      // word was entered
      else
         cout << "<p>Your word is: " << wordString << "</p>";
   }
   
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
