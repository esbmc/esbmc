// Pins the add arm of pointer_offset2t::do_simplify at the irep2
// level, independent of frontend lowering: with a namespace installed,
// the offset over a named-struct pointer must fold to the scaled form
// instead of bailing to nil. The end-to-end pin for the same arm lives
// in regression/esbmc/pointer_offset_struct_arith.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

TEST_CASE("pointer_offset over add2t folds for named structs", "[irep2]")
{
  // A two-member struct registered in a namespace under a tag, so the
  // symbol type resolves the way a frontend-declared struct does.
  contextt ctx;
  std::vector<type2tc> members = {get_int32_type(), get_int32_type()};
  std::vector<irep_idt> names = {"a", "b"};
  std::vector<irep_idt> pretty = {"a", "b"};
  type2tc strct = struct_type2tc(members, names, pretty, "tag-node");
  symbolt sym;
  sym.id = "tag-node";
  sym.name = "tag-node";
  sym.is_type = true;
  sym.set_type(strct);
  ctx.add(sym);
  namespacet ns(ctx);
  migrate_namespace_lookup = &ns;

  type2tc symtype = symbol_type2tc("tag-node");
  type2tc ptrtype = pointer_type2tc(symtype);
  expr2tc p = symbol2tc(ptrtype, "p");
  expr2tc two = constant_int2tc(get_int32_type(), BigInt(2));
  expr2tc arith = add2tc(ptrtype, p, two);
  expr2tc off = pointer_offset2tc(get_int64_type(), arith);
  expr2tc s = off->simplify();

  migrate_namespace_lookup = nullptr;

  // pointer_offset(p) + 2*sizeof(struct{int;int;}) — the fold must
  // produce the scaled form, not bail to nil.
  REQUIRE(!is_nil_expr(s));
  REQUIRE(is_add2t(s));
  const add2t &ad = to_add2t(s);
  REQUIRE(is_constant_int2t(ad.side_2));
  REQUIRE(to_constant_int2t(ad.side_2).value == 16);
}
