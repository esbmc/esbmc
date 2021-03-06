/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cstring>
#include <util/message/message_stream.h>

void message_streamt::error_parse_line(
  VerbosityLevel level,
  const std::string &line)
{
  std::string error_msg = line;

  const char *tptr = line.c_str();

  if(strncmp(tptr, "file ", 5) == 0)
  {
    int state = 0;
    std::string file, line_no, column, _error_msg, function;

    tptr += 5;

    char previous = 0;

    while(*tptr != 0)
    {
      if(strncmp(tptr, " line ", 6) == 0 && state != 4)
      {
        state = 1;
        tptr += 6;
        continue;
      }
      if(strncmp(tptr, " column ", 8) == 0 && state != 4)
      {
        state = 2;
        tptr += 8;
        continue;
      }
      else if(strncmp(tptr, " function ", 10) == 0 && state != 4)
      {
        state = 3;
        tptr += 10;
        continue;
      }
      else if(*tptr == ':' && state != 4)
      {
        if(tptr[1] == ' ' && previous != ':')
        {
          state = 4;
          tptr++;
          while(*tptr == ' ')
            tptr++;
          continue;
        }
      }

      if(state == 0) // file
        file += *tptr;
      else if(state == 1) // line number
        line_no += *tptr;
      else if(state == 2) // column
        column += *tptr;
      else if(state == 3) // function
        function += *tptr;
      else if(state == 4) // error message
        _error_msg += *tptr;

      previous = *tptr;

      tptr++;
    }

    if(state == 4)
    {
      saved_error_location.set_line(line_no);
      saved_error_location.set_file(file);
      saved_error_location.set_column(column);
      error_msg = _error_msg;
      saved_error_location.set_function(function);
    }
  }
  else
  {
    int state = 0;
    std::string file, line_no;

    while(*tptr != 0)
    {
      if(state == 0)
      {
        if(*tptr == ':')
          state++;
        else
          file += *tptr;
      }
      else if(state == 1)
      {
        if(*tptr == ':')
          state++;
        else if(isdigit(*tptr))
          line_no += *tptr;
        else
          state = 3;
      }

      tptr++;
    }

    if(state == 2)
    {
      saved_error_location.set_line(line_no);
      saved_error_location.set_file(file);
      saved_error_location.set_function("");
      saved_error_location.set_column("");
    }
  }

  message_handler.print(level, error_msg, saved_error_location);
}

void message_streamt::error_parse(
  VerbosityLevel level,
  const std::string &error)
{
  const char *tptr = error.c_str();

  std::string line;

  while(true)
  {
    switch(*tptr)
    {
    case 0:
      return;
    case '\n':
      error_parse_line(level, line);
      line.clear();
      break;

    case '\r':
      break;
    default:
      line += *tptr;
    }

    tptr++;
  }
}
