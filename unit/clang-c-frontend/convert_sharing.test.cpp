/*******************************************************************\
Module: Unit tests for node sharing in clang_c_convertert

 The converter used to build a fresh type and fresh location fields for
 every AST node, which made converting a large program cost many times
 clang's own AST (#10190). Equal values must now share one node.
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <clang-c-frontend/AST/vfs_paths.h>
#include <langapi/language_ui.h>
#include <util/base/filesystem.h>
#include <util/expr/expr_util.h>
#include <util/lang/c_types.h>
#include <util/symtab/fix_symbol.h>

#include <vector>

static const char source[] =
  "int a, b;\n"
  "int f(void) { return (a == 1) && (b == 2); }\n";

static void
collect(const irept &e, const irep_idt &id, std::vector<const exprt *> &out)
{
  if (e.id() == id)
    out.push_back(static_cast<const exprt *>(&e));
  for (const irept &op : e.get_sub())
    collect(op, id, out);
  for (const auto &[name, sub] : e.get_named_sub())
    collect(sub, id, out);
}

// Two ireps share storage exactly when they read the same node.
static bool same_node(const irept &x, const irept &y)
{
  return &x.get_sub() == &y.get_sub();
}

TEST_CASE("the C converter shares equal types and locations", "[core]")
{
  cmdlinet cmd = goto_factory::get_default_cmdline("");
  optionst opts = goto_factory::get_default_options(cmd);
  goto_factory::config_environment(
    goto_factory::Architecture::BIT_64, cmd, opts);

  std::string path = clang_vfs_root() + "/unit/clang-c-frontend/sharing.c";
  file_operations::filesystemt::get().add_bundled(
    path, source, sizeof(source) - 1);

  language_uit l;
  REQUIRE(!l.parse(path));
  REQUIRE(!l.typecheck());
  const symbolt *f = l.context.find_symbol("c:@F@f");
  REQUIRE(f != nullptr);

  std::vector<const exprt *> constants;
  collect(f->get_value(), "constant", constants);
  const exprt *one = nullptr, *two = nullptr;
  for (const exprt *c : constants)
    if (c->get("#cformat") == "1")
      one = c;
    else if (c->get("#cformat") == "2")
      two = c;
  REQUIRE(one != nullptr);
  REQUIRE(two != nullptr);

  // Two int literals share one type node.
  REQUIRE(one->type() == two->type());
  REQUIRE(same_node(one->type(), two->type()));

  // Two comparisons share one bool type node, still after linking.
  std::vector<const exprt *> equalities;
  collect(f->get_value(), "=", equalities);
  REQUIRE(equalities.size() == 2);
  REQUIRE(equalities[0]->type().id() == "bool");
  REQUIRE(same_node(equalities[0]->type(), equalities[1]->type()));

  // Two locations on one line share every field but the column.
  const locationt &x = one->location();
  const locationt &y = two->location();
  REQUIRE(x.get_column() != y.get_column());
  for (const char *field : {"file", "line", "function"})
  {
    INFO(field);
    REQUIRE(x.find(field) == y.find(field));
    REQUIRE(same_node(x.find(field), y.find(field)));
  }
}

TEST_CASE("linking leaves a value alone when nothing is renamed", "[core]")
{
  symbolt symbol;
  symbol.id = "c:@x";
  symbol.set_value(gen_zero(int_type()));
  const exprt before = symbol.get_value();

  fix_symbolt fixer;
  fixer.fix_symbol(symbol);

  REQUIRE(symbol.get_value() == before);
  REQUIRE(same_node(symbol.get_value(), before));
}
