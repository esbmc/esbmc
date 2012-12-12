// Fig. 16.12: post.cpp
// Demonstrates POST method with XHTML form.
#include <iostream>

using std::cout;
using std::cin;

#include <string>

using std::string;

#include <cstdlib>

int main()
{
   char postString[ 1024 ] = ""; // variable to hold query string
   string dataString = "";
   string nameString = "";
   string wordString = "";
   int contentLength = 0;
   
   // content was submitted
   if ( getenv( "CONTENT_LENGTH" ) )
      contentLength = atoi( getenv( "CONTENT_LENGTH" ) );
   
      cin.read( postString, contentLength );
      dataString = postString;

   // output header
   cout << "Content-type: text/html\n\n";
   
   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>"
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\" \"http://www.w3.org/TR/xhtml1"
        << "/DTD/xhtml1-transitional.dtd\">";

   // output html element and some of its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Using POST with Forms</title></head>"
        << "<body>";
   
   // output xhtml form
   cout << "<p>Enter one of your favorite words here:</p>"
        << "<form method = \"post\" action = \"post.cgi\">"
        << "<input type = \"text\" name = \"word\" />"
        << "<input type = \"submit\" value = \"Submit Word\" />"
        << "</form>";
   
   // data was sent using POST
   if ( contentLength > 0 ) {
      int nameLocation = 
         dataString.find_first_of( "word=" ) + 5;
      
      int endLocation = dataString.find_first_of( "&" ) - 1;
      
      // retrieve entered word
      wordString = dataString.substr( nameLocation,
         endLocation - nameLocation );
      
      // no data was entered in text field
      if ( wordString == "" )
         cout << "<p>Please enter a word.</p>";
      
      // output word
      else
         cout << "<p>Your word is: " << wordString << "</p>";
   
   } // end if
   
   // no data was sent
   else
      cout << "<p>Please enter a word.</p>";
   
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
