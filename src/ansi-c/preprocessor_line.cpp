/*******************************************************************\

Module: ANSI-C Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/preprocessor_line.h>
#include <ansi-c/unescape_string.h>
#include <cctype>
#include <cstdlib>

void preprocessor_line(
  const char *text,
  unsigned &line_no,
  irep_idt &file_name)
{
  const char *ptr=text;
  std::string line_number;
  
  // skip WS
  while(*ptr==' ' || *ptr=='\t') ptr++;

  // skip #
  if(*ptr!='#') return;
  ptr++;

  // skip WS
  while(*ptr==' ' || *ptr=='\t') ptr++;

  // skip "line"
  if(*ptr=='l')
  {
    while(*ptr!=0 && *ptr!=' ' && *ptr!='\t') ptr++;
  }

  // skip WS
  while(*ptr==' ' || *ptr=='\t') ptr++;

  // get line number
  while(isdigit(*ptr))
  {
    line_number+=*ptr;
    ptr++;
  }
  
  // skip until "
  while(*ptr!='\n' && *ptr!='"') ptr++;

  line_no=atoi(line_number.c_str());

  // skip "
  if(*ptr!='"')
    return;
  
  ptr++;
  
  std::string file_name_tmp;

  // get file name
  while(*ptr!='\n' && *ptr!='"')
  {
    file_name_tmp+=*ptr;
    ptr++;
  }

  std::string file_name_tmp2;
  unescape_string(file_name_tmp, file_name_tmp2);
  file_name=file_name_tmp2;
}
