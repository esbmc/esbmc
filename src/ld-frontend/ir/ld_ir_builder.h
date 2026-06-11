#pragma once

#include <ld-frontend/parser/ld_ast.h>
#include <ld-frontend/ir/ld_ir.h>

// Lowers a type-checked LdAst into LdIR by applying SOS rule assignments.
class LdIRBuilder
{
public:
  LdIR build(const LdAst &ast);

private:
  LdIRNode lower_element(const RungElement &elem);
  LdIRRung lower_rung(const RungNode &rung);
};
