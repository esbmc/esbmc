//
// Created by rafaelsamenezes on 23/09/2021.
//

#include <jimple-frontend/jimple-language.h>
#include <jimple-frontend/jimple-converter.h>
#include <util/message/format.h>

bool jimple_languaget::typecheck(
  contextt &context,
  const std::string &,
  const messaget &msg)
{
  msg.status(
    fmt::format("Converting Jimple module {} to GOTO", root.get_class_name()));

  contextt new_context(msg);
  jimple_converter converter(context, root, msg);
  if(converter.convert())
  {
    msg.error(
      fmt::format("Failed to convert module {}", root.get_class_name()));
    return true;
  }

  return false;
}