
#include "c_types.h"
#include "std_expr.h"
#include <util/linker_script.h>
#include <util/expr_util.h>
#include <util/arith_tools.h>
#include <big-int/bigint.hh>
linker_script::linker_script(const std::string &ld_script_src)
{
}

std::optional<linker_script::script_t>
linker_script::parse(const std::string &ld_script_src)
{
  generate_example();
  return {};
}

bool linker_script::apply(contextt &program)
{
  symbolt new_symbol;
  code_typet main_type;
  main_type.return_type() = empty_typet();
  new_symbol.id = "__ESBMC_linker_assumptions";
  new_symbol.name = "__ESBMC_linker_assumptions";
  new_symbol.type.swap(main_type);

  code_blockt init_code;
  init_code.make_block();

  program.foreach_operand(
    [this, &init_code](const symbolt &sym)
    {
      if (!sym.type.get("section").empty())
      {
        layout.sections[sym.type.get("section").as_string()]->symbols.push_back(
          sym);
        exprt assumption = layout.sections[sym.type.get("section").as_string()]
                             ->memory_region->to_range_assumption(sym);
        init_code.move_to_operands(assumption);
      }
    });

  for (auto &[k, v] : layout.sections)
  {
    exprt assumption = v->to_ordering_assumption();   
    init_code.move_to_operands(assumption);
  }
  new_symbol.value.swap(init_code);
  program.move(new_symbol);
  return false;
}

exprt linker_script::output_section_t::to_ordering_assumption() const
{
  exprt conj = gen_boolean(true);

  if (symbols.empty() || symbols.size() == 1)
    return code_assumet(conj);

  for (size_t i = 1; i < symbols.size(); i++)
    {

    typecast_exprt address1_value(
      address_of_exprt(symbol_expr(symbols[i - 1])), uint_type());
    typecast_exprt address2_value(
				  address_of_exprt(symbol_expr(symbols[i])), uint_type());

    // TODO: get symbol length
    unsigned symbol1_size = atoi(symbols[i-1].type.width().c_str());
    exprt operation("+", uint_type());
    operation.copy_to_operands(address1_value, from_integer(symbol1_size, uint_type()));

    // &symbol1  + symbol1_size <= &symbol2
    exprt ordering("<", bool_type());
    ordering.copy_to_operands(operation, address2_value);

    conj = and_exprt(conj, ordering);

  }

  return code_assumet(conj);
}

exprt linker_script::memory_region_t::to_range_assumption(
  const symbolt &sym) const
{
  typecast_exprt address_value(address_of_exprt(symbol_expr(sym)), uint_type());
  BigInt origin_value(origin);
  // TODO: get symbol length
  unsigned symbol_size = atoi(sym.type.width().c_str());


  // Range
  BigInt max_address(origin + length + symbol_size);

  // origin <= address_value < origin + length - sizeof(sym)
  exprt side1("<=", bool_type());
  side1.copy_to_operands(
    from_integer(origin_value, uint_type()), address_value);

  exprt side2("<", bool_type());
  side2.copy_to_operands(address_value, from_integer(max_address, uint_type()));

  exprt conjunction = and_exprt(side1, side2);
  return code_assumet(conjunction);
}

void linker_script::generate_example()
{
  /*
    
  MEMORY {
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K;
  SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K;
}

SECTIONS {
  .text        : { *(.text .text.*) } > FLASH
  .rodata      : { *(.rodata .rodata.*) } > FLASH
  .data        : { *(.data .data.*) } > SRAM
  .bss         : { *(.bss .bss.* COMMON) } > SRAM
  .fcu 0x20001000 ALIGN(8) : { *(.fcu) } > SRAM
}
   */

  // FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512;
  std::shared_ptr<memory_region_t> flash = std::make_shared<memory_region_t>(
    memory_region_t({0x08000000, 512 * 1024, memory_region_t::RX}));
  // SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K;
  std::shared_ptr<memory_region_t> sram = std::make_shared<memory_region_t>(
    memory_region_t({0x08000000, 512 * 1024, memory_region_t::RWX}));

  std::shared_ptr<output_section_t> fcu = std::make_shared<output_section_t>(
    output_section_t({{0x20001000}, {8}, {".fcu"}, sram}));

  layout.memory["FLASH"] = flash;
  layout.memory["SRAM"] = sram;
  layout.sections[".fcu"] = fcu;
}
