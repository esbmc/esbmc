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

#include <vector>

static const char source[] =
  "int a, b;\n"
  "int f(void) { return (a == 1) && (b == 2); }\n";

static void collect_constants(const irept &e, std::vector<const exprt *> &out)
{
  if (e.id() == "constant")
    out.push_back(static_cast<const exprt *>(&e));
  for (const irept &op : e.get_sub())
    collect_constants(op, out);
  for (const auto &[name, sub] : e.get_named_sub())
    collect_constants(sub, out);
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
  collect_constants(f->get_value(), constants);
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
