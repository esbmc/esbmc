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

static void collect_constants(const irept &e, std::vector<const exprt *> &out)
{
  collect(e, "constant", out);
}

// Two ireps share storage exactly when they read the same node.
static bool same_node(const irept &x, const irept &y)
{
  return &x.get_sub() == &y.get_sub();
}

// Parse and typecheck `src` and return the body of its function f.
template <size_t N>
static exprt
typecheck_f(language_uit &l, const char *name, const char (&src)[N])
{
  cmdlinet cmd = goto_factory::get_default_cmdline("");
  optionst opts = goto_factory::get_default_options(cmd);
  goto_factory::config_environment(
    goto_factory::Architecture::BIT_64, cmd, opts);

  std::string path = clang_vfs_root() + "/unit/clang-c-frontend/" + name;
  file_operations::filesystemt::get().add_bundled(path, src, N - 1);

  REQUIRE(!l.parse(path));
  REQUIRE(!l.typecheck());
  const symbolt *f = l.context.find_symbol("c:@F@f");
  REQUIRE(f != nullptr);
  return f->get_value();
}

TEST_CASE("the C converter shares equal types and locations", "[core]")
{
  language_uit l;
  const exprt body = typecheck_f(l, "sharing.c", source);

  std::vector<const exprt *> constants;
  collect_constants(body, constants);
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

static const char column_source[] =
  "int a, b;\n"
  "int g(int);\n"
  "int f(void) {\n"
  "  if (a == 1) return g(1);\n"
  "  if (b == 2) return g(2);\n"
  "  return 0;\n"
  "}\n";

TEST_CASE("equal columns and types share one node after linking", "[core]")
{
  language_uit l;
  const exprt body = typecheck_f(l, "column_sharing.c", column_source);

  std::vector<const exprt *> eqs;
  collect(body, "=", eqs);
  REQUIRE(eqs.size() == 2);
  const exprt &x = *eqs[0], &y = *eqs[1];

  // The two comparisons sit on different lines at the same column.
  REQUIRE(x.location().get_line() != y.location().get_line());
  REQUIRE(x.location().find("column") == y.location().find("column"));
  REQUIRE(same_node(x.location().find("column"), y.location().find("column")));

  // Both comparisons are typed bool, through clang_c_adjust and linking.
  REQUIRE(x.type().is_bool());
  REQUIRE(same_node(x.type(), y.type()));

  // Linking with nothing to rename leaves the callee's code type shared.
  std::vector<const exprt *> calls;
  collect(body, "sideeffect", calls);
  REQUIRE(calls.size() == 2);
  const typet &gx = calls[0]->op0().type(), &gy = calls[1]->op0().type();
  REQUIRE(gx.is_code());
  REQUIRE(same_node(gx, gy));
}
