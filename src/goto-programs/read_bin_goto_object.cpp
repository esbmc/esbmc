/*******************************************************************\

Module: Read goto object files.

Author: CM Wintersteiger

Date: June 2006

\*******************************************************************/

#include <goto-programs/goto_function_serialization.h>
#include <goto-programs/goto_program_irep.h>
#include <goto-programs/read_bin_goto_object.h>
#include <langapi/mode.h>
#include <util/base_type.h>
#include <util/irep_serialization.h>
#include <util/message_stream.h>
#include <util/namespace.h>
#include <util/symbol_serialization.h>

#define BINARY_VERSION 1

bool read_bin_goto_object(
  std::istream &in,
  const std::string &filename,
  contextt &context,
  goto_functionst &functions,
  message_handlert &message_handler)
{
  message_streamt message_stream(message_handler);

  {
    char hdr[4];
    hdr[0] = in.get();
    hdr[1] = in.get();
    hdr[2] = in.get();

    if(hdr[0] != 'G' || hdr[1] != 'B' || hdr[2] != 'F')
    {
      hdr[3] = in.get();

      if(hdr[0] == 0x7f && hdr[1] == 'E' && hdr[2] == 'L' && hdr[3] == 'F')
      {
        if(filename != "")
          message_stream.str << "Sorry, but I can't read ELF binary `"
                             << filename << "'";
        else
          message_stream.str << "Sorry, but I can't read ELF binaries";
      }
      else
        message_stream.str << "`" << filename << "' is not a goto-binary."
                           << std::endl;

      message_stream.error();

      return false;
    }
  }

  irep_serializationt::ireps_containert ic;
  irep_serializationt irepconverter(ic);
  symbol_serializationt symbolconverter(ic);
  goto_function_serializationt gfconverter(ic);

  {
    unsigned version = irepconverter.read_long(in);

    if(version != BINARY_VERSION)
    {
      message_stream.str
        << "The input was compiled with a different version of "
        << "goto-cc, please recompile";
      message_stream.warning();
      return false;
    }
  }

  unsigned count = irepconverter.read_long(in);

  for(unsigned i = 0; i < count; i++)
  {
    irept t;
    symbolconverter.convert(in, t);
    symbolt symbol;
    symbol.from_irep(t);

    if(!symbol.is_type && symbol.type.is_code())
    {
      // makes sure there is an empty function
      // for every function symbol and fixes
      // the function types.
      functions.function_map[symbol.name].type = to_code_type(symbol.type);
    }
    context.add(symbol);
  }

  count = irepconverter.read_long(in);
  for(unsigned i = 0; i < count; i++)
  {
    irept t;
    dstring fname = irepconverter.read_string(in);
    gfconverter.convert(in, t);
    goto_functiont &f = functions.function_map[fname];
    convert(t, f.body);
    f.body_available = f.body.instructions.size() > 0;
  }

  return false;
}
