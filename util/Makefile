SRC = arith_tools.cpp base_type.cpp cmdline.cpp config.cpp context.cpp \
      expr.cpp expr_util.cpp i2string.cpp irep.cpp language.cpp \
      lispexpr.cpp lispirep.cpp location.cpp message.cpp language_file.cpp \
      mp_arith.cpp namespace.cpp parseoptions.cpp rename.cpp \
      replace_expr.cpp threeval.cpp typecheck.cpp graph.cpp \
      type.cpp cnf_simplify.cpp str_getline.cpp strstream2string.cpp \
      bitvector.cpp parser.cpp map_util.cpp replace_symbol.cpp actuals.cpp \
      get_module.cpp string_hash.cpp string_container.cpp identifier.cpp \
      rational.cpp options.cpp c_misc.cpp simplify_expr.cpp dstring.cpp \
      find_symbols.cpp rational_tools.cpp ui_message.cpp simplify_utils.cpp \
      string2array.cpp time_stopping.cpp symbol.cpp expr_visitor.cpp \
      type_eq.cpp guard.cpp array_name.cpp gcd.cpp message_stream.cpp \
      substitute.cpp decision_procedure.cpp union_find.cpp pretty_names.cpp \
      xml.cpp xml_irep.cpp std_types.cpp std_code.cpp pointer_offset_size.cpp \
      format_constant.cpp find_macros.cpp ref_expr_set.cpp std_expr.cpp \
      irep_serialization.cpp symbol_serialization.cpp fixedbv.cpp \
      pointer_arithmetic.cpp ieee_float.cpp signal_catcher.cpp \
      bv_arithmetic.cpp tempdir.cpp var_name.cpp

OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: util.o ieee_float_test

include ../config.inc
include ../common

ifdef MODULE_FLOATBV
  CXXFLAGS += -DHAVE_FLOATBV
endif

###############################################################################

util.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o util.o $(OBJ)

clean:
	rm -f *.o ieee_float_test
	-rm *.d

ieee_float_test: ieee_float_test.o ieee_float.o util.o
	$(CXX) $(CXXFLAGS) -o ieee_float_test util.o ieee_float_test.o ../big-int/bigint.o
