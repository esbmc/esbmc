/*******************************************************************\

Module: ANSI-C Misc Utilities

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cstdio>
#include <util/c_misc.h>

void MetaChar(std::string &out, char c, bool inString)
{
  switch (c)
  {
    case '\'':
      if (inString) 
        out+="'";
      else
        out+="\\'";
      break;

    case '"':
      if (inString) 
        out+="\\\"";
      else
        out+="\"";
      break;

    case '\0':
      out+="\\0";
      break;

    case '\\':
      out+="\\\\";
      break;

    case '\n':
      out+="\\n";
      break;

    case '\t':
      out+="\\t";
      break;

    case '\r':
      out+="\\r";
      break;

    case '\f':
      out+="\\f";
      break;

    case '\b':
      out+="\\b";
      break;

    case '\v':
      out+="\\v";
      break;

    case '\a':
      out+="\\a";
      break;

    default:
      // Show low and high ascii as octal
      if ((c < ' ') || (c >= 127))
      {
          char octbuf[8];
          sprintf(octbuf, "%03o", (unsigned char) c);
          out+="\\";
          out+=octbuf;
      }
      else
          out+=c;
      break;
  }
}

void MetaString(std::string &out, const std::string &in)
{
  for(char i : in)
    MetaChar(out, i, true);
}
