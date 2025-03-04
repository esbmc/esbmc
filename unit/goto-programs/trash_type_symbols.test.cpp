#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
// Provides `mode_table` needed transitively by `goto_convert_functions.h`
#include "../testing-utils/goto_factory.h"
#include <goto-programs/goto_convert_functions.h>
#include <util/context.h>
#include <util/symbol.h>

static bool is_complete(goto_convert_functionst &converter, typet type)
{
  return !converter.ensure_type_is_complete(type);
}

static void setup_type(
  contextt &context,
  const std::string &name,
  const struct_typet &struct_type)
{
  symbolt symbol;
  symbol.id = "tag-" + name;
  symbol.name = name;
  symbol.type = struct_type;
  symbol.is_type = true;
  context.add(symbol);
}
TEST_CASE("trash_type_symbols", "[goto_convert_functions]")
{
  contextt context;
  optionst options;
  goto_functionst functions;
  goto_convert_functionst converter(context, options, functions);

  SECTION("complete type")
  {
    // Create a complete type
    struct_typet struct_type;
    setup_type(context, "complete_type", struct_type);

    const symbolt *resolved_symbol = context.find_symbol("tag-complete_type");
    REQUIRE(resolved_symbol);
    REQUIRE(resolved_symbol->type.id() == "struct");
    REQUIRE(is_complete(converter, resolved_symbol->type));

    converter.thrash_type_symbols();

    // Verify that the type has been unchanged
    REQUIRE(resolved_symbol);
    REQUIRE(resolved_symbol->type.id() == "struct");
    REQUIRE(is_complete(converter, resolved_symbol->type));
  }

  SECTION("complete type with symbolic component")
  {
    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.components().emplace_back("float_component", floatbv_typet());
      setup_type(context, "complete_type_2", struct_type);
    }

    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.components().emplace_back(
        "symbolic_component", symbol_typet("tag-complete_type_2"));
      setup_type(context, "complete_type", struct_type);
    }

    // Verify that the type has been resolved correctly
    const symbolt *resolved_symbol = context.find_symbol("tag-complete_type");
    REQUIRE(resolved_symbol);
    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is incomplete because it contains a symbolic component
    REQUIRE(!is_complete(converter, resolved_symbol->type));

    converter.thrash_type_symbols();

    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type should now be complete
    REQUIRE(is_complete(converter, resolved_symbol->type));
  }

  SECTION("complete type with symbolic component via pointer")
  {
    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.components().emplace_back("float_component", floatbv_typet());
      setup_type(context, "complete_type_2", struct_type);
    }

    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.components().emplace_back(
        "symbolic_component_via_pointer",
        pointer_typet(symbol_typet("tag-complete_type_2")));
      setup_type(context, "complete_type", struct_type);
    }

    // Verify that the type has been resolved correctly
    const symbolt *resolved_symbol = context.find_symbol("tag-complete_type");
    REQUIRE(resolved_symbol);
    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is complete because pointer types are always complete
    REQUIRE(is_complete(converter, resolved_symbol->type));

    converter.thrash_type_symbols();

    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is still complete because pointer types are always complete
    REQUIRE(is_complete(converter, resolved_symbol->type));
  }

  SECTION(
    "thrash_type_symbols complete type with symbolic component via pointer "
    "mutually recursive")
  {
    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.components().emplace_back(
        "symbolic_component_via_pointer",
        pointer_typet(symbol_typet("tag-complete_type")));
      setup_type(context, "complete_type_2", struct_type);
    }

    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.components().emplace_back(
        "symbolic_component_via_pointer",
        pointer_typet(symbol_typet("tag-complete_type_2")));
      setup_type(context, "complete_type", struct_type);
    }

    auto verify_pre = [&](const std::string &id) {
      // Verify that the type has been resolved correctly
      const symbolt *resolved_symbol = context.find_symbol(id);
      REQUIRE(resolved_symbol);
      REQUIRE(resolved_symbol->type.id() == "struct");
      // The type is complete because pointer types are always complete
      REQUIRE(is_complete(converter, resolved_symbol->type));
    };
    verify_pre("tag-complete_type");
    verify_pre("tag-complete_type_2");

    converter.thrash_type_symbols();

    auto verify_post = [&](const std::string &id) {
      // Verify that the type has been resolved correctly
      const symbolt *resolved_symbol = context.find_symbol(id);
      REQUIRE(resolved_symbol);
      REQUIRE(resolved_symbol->type.id() == "struct");
      // The type is still complete because pointer types are always complete
      REQUIRE(is_complete(converter, resolved_symbol->type));
    };

    verify_post("tag-complete_type");
    verify_post("tag-complete_type_2");
  }

  SECTION("complete type with symbolic component via pointer recursive")
  {
    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.components().emplace_back(
        "symbolic_component_via_pointer",
        pointer_typet(symbol_typet("tag-complete_type")));
      setup_type(context, "complete_type", struct_type);
    }

    // Verify that the type has been resolved correctly
    const symbolt *resolved_symbol = context.find_symbol("tag-complete_type");
    REQUIRE(resolved_symbol);
    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is complete because pointer types are always complete
    REQUIRE(is_complete(converter, resolved_symbol->type));

    converter.thrash_type_symbols();

    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is still complete because pointer types are always complete
    REQUIRE(is_complete(converter, resolved_symbol->type));
  }

  SECTION("complete type with symbolic return type")
  {
    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.methods().emplace_back(
        "symbolic_method", code_typet({}, symbol_typet("tag-complete_type")));
      setup_type(context, "complete_type", struct_type);
    }

    // Verify that the type has been resolved correctly
    const symbolt *resolved_symbol = context.find_symbol("tag-complete_type");
    REQUIRE(resolved_symbol);
    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is complete because it contains no components
    REQUIRE(is_complete(converter, resolved_symbol->type));
    // Verify that the method's return type is incomplete
    REQUIRE(!is_complete(
      converter,
      to_code_type(
        to_struct_type(resolved_symbol->type).methods().front().type())
        .return_type()));

    converter.thrash_type_symbols();

    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is still complete because it contains no components
    REQUIRE(is_complete(converter, resolved_symbol->type));
    // Verify that the method's return type is still incomplete, because apparently we don't resolve return types of methods
    REQUIRE(!is_complete(
      converter,
      to_code_type(
        to_struct_type(resolved_symbol->type).methods().front().type())
        .return_type()));
  }

  SECTION("complete type with symbolic argument type")
  {
    {
      // Create a complete type
      struct_typet struct_type;
      struct_type.methods().emplace_back(
        "symbolic_method",
        code_typet(
          code_typet::argumentst{symbol_typet("tag-complete_type")},
          empty_typet()));
      setup_type(context, "complete_type", struct_type);
    }

    // Verify that the type has been resolved correctly
    const symbolt *resolved_symbol = context.find_symbol("tag-complete_type");
    REQUIRE(resolved_symbol);
    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is complete because it contains no components
    REQUIRE(is_complete(converter, resolved_symbol->type));
    // Verify that the method's argument type is incomplete
    REQUIRE(!is_complete(
      converter,
      to_code_type(
        to_struct_type(resolved_symbol->type).methods().front().type())
        .arguments()
        .front()
        .type()));

    converter.thrash_type_symbols();

    REQUIRE(resolved_symbol->type.id() == "struct");
    // The type is still complete because it contains no components
    REQUIRE(is_complete(converter, resolved_symbol->type));
    // Verify that the method's argument type is now complete
    REQUIRE(is_complete(
      converter,
      to_code_type(
        to_struct_type(resolved_symbol->type).methods().front().type())
        .arguments()
        .front()
        .type()));
  }
}
