#pragma once

#include <irep2/irep2_expr.h>

#include <map>
#include <string>

namespace llm
{
/// Printed name -> symbol. The parser resolves names only through it.
using scopet = std::map<std::string, expr2tc>;
} // namespace llm
