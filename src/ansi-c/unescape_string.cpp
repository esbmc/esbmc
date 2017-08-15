/*******************************************************************\

Module: ANSI-C Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/unescape_string.h>
#include <cctype>
#include <cstdio>

void unescape_string(const std::string &src, std::string &dest)
{
  dest="";
  dest.reserve(src.size());

  for(unsigned i=0; i<src.size(); i++)
  {
    char ch=src[i];

    if(ch=='\\')
    {
      i++;
      ch=src[i];
      switch(ch)
      {
      case '\\': dest+=ch; break;
      case 'n': dest+='\n'; break; /* NL (0x0a) */
      case 't': dest+='\t'; break; /* HT (0x09) */
      case 'v': dest+='\v'; break; /* VT (0x0b) */
      case 'b': dest+='\b'; break; /* BS (0x08) */
      case 'r': dest+='\r'; break; /* CR (0x0d) */
      case 'f': dest+='\f'; break; /* FF (0x0c) */
      case 'a': dest+='\a'; break; /* BEL (0x07) */
      case '"': dest.push_back('"'); break;
      case '\'': dest.push_back('\''); break;

      case 'x': // hex
        i++;

        {
          std::string hex;
          while(isxdigit(src[i]))
          {
            i++;
            hex+=src[i];
          }

          unsigned int result;
          sscanf(hex.c_str(), "%x", &result);
          ch=result;
        }

        dest+=ch;

        break;

      default:
        if(isdigit(ch)) // octal
        {
          std::string octal;

          while(isdigit(src[i]))
          {
            octal+=src[i];
            i++;
          }

          unsigned int result;
          sscanf(octal.c_str(), "%o", &result);
          ch=result;
          dest+=ch;
        }
        else
        {
          dest+='\\';
          dest+=ch;
        }
      }
    }
    else
      dest+=ch;
  }
}
