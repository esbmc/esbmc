#pragma once

#include <llm/goto/scope.h>
#include <util/symtab/namespace.h>

#include <optional>
#include <string>

namespace llm
{
/// Parse a model-written C expression, resolving names only through @p scope
/// and converting operands as C does. Returns nullopt on anything that does
/// not parse or type-check.
std::optional<expr2tc>
parse_expr(const std::string &text, const scopet &scope, const namespacet &ns);
} // namespace llm
