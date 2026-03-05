#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/python_converter.h>
#include <python-frontend/global_scope.h>
#include <python-frontend/type_handler.h>
#include <util/context.h>

TEST_CASE("complex type helpers", "[python-frontend][complex]")
{
  SECTION("detects complex struct type")
  {
    const struct_typet complex_type = get_complex_struct_type();

    REQUIRE(is_complex_type(complex_type));
    REQUIRE(complex_type.components().size() == 2);
    REQUIRE(complex_type.components()[0].get_name() == "real");
    REQUIRE(complex_type.components()[1].get_name() == "imag");
    REQUIRE(complex_type.components()[0].type() == double_type());
    REQUIRE(complex_type.components()[1].type() == double_type());
  }

  SECTION("builds complex expression with double components")
  {
    const exprt real = from_integer(3, signedbv_typet(32));
    const exprt imag = from_double(4.5, double_type());

    const exprt complex_expr = make_complex(real, imag);

    REQUIRE(complex_expr.id() == "struct");
    REQUIRE(is_complex_type(complex_expr.type()));
    REQUIRE(complex_expr.operands().size() == 2);
    REQUIRE(complex_expr.operands()[0].type() == double_type());
    REQUIRE(complex_expr.operands()[1].type() == double_type());
  }

  SECTION("promotes scalar and keeps complex unchanged")
  {
    const exprt scalar = from_integer(7, signedbv_typet(32));
    const exprt promoted = promote_to_complex(scalar);

    REQUIRE(is_complex_type(promoted.type()));
    REQUIRE(promoted.operands().size() == 2);
    REQUIRE(promoted.operands()[0].type() == double_type());
    REQUIRE(promoted.operands()[1].type() == double_type());

    const exprt already_complex =
      make_complex(from_double(1.0, double_type()), from_double(2.0, double_type()));
    const exprt unchanged = promote_to_complex(already_complex);

    REQUIRE(unchanged == already_complex);
  }

  SECTION("recognizes symbol type representation")
  {
    const typet complex_symbol_type = symbol_typet("tag-complex");
    REQUIRE(is_complex_type(complex_symbol_type));
  }

  SECTION("registers complex type in symbol table through get_typet")
  {
    contextt context;
    global_scope global_scope;
    const nlohmann::json ast = {
      {"_type", "Module"},
      {"body", nlohmann::json::array()},
      {"filename", "test.py"},
      {"type_ignores", nlohmann::json::array()}};

    python_converter converter(context, &ast, global_scope);
    const typet complex_type =
      converter.get_type_handler().get_typet(std::string("complex"));

    REQUIRE(is_complex_type(complex_type));

    const symbolt *complex_type_symbol = context.find_symbol("tag-complex");
    REQUIRE(complex_type_symbol != nullptr);
    REQUIRE(complex_type_symbol->is_type);
    REQUIRE(is_complex_type(complex_type_symbol->type));
  }
}
