ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
MAINOBJ = $(OBJDIR)/util.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= arith_tools.cpp base_type.cpp cmdline.cpp config.cpp context.cpp \
      expr.cpp expr_util.cpp i2string.cpp irep.cpp language.cpp \
      location.cpp message.cpp language_file.cpp \
      mp_arith.cpp namespace.cpp parseoptions.cpp rename.cpp \
      replace_expr.cpp threeval.cpp typecheck.cpp \
      type.cpp str_getline.cpp strstream2string.cpp \
      bitvector.cpp parser.cpp replace_symbol.cpp \
      string_hash.cpp string_container.cpp identifier.cpp \
      options.cpp c_misc.cpp simplify_expr.cpp dstring.cpp \
      find_symbols.cpp ui_message.cpp simplify_utils.cpp \
      string2array.cpp time_stopping.cpp symbol.cpp \
      type_eq.cpp guard.cpp array_name.cpp message_stream.cpp \
      decision_procedure.cpp union_find.cpp pretty_names.cpp \
      xml.cpp xml_irep.cpp std_types.cpp std_code.cpp pointer_offset_size.cpp \
      format_constant.cpp std_expr.cpp \
      irep_serialization.cpp symbol_serialization.cpp fixedbv.cpp \
      pointer_arithmetic.cpp ieee_float.cpp signal_catcher.cpp \
      bv_arithmetic.cpp irep2.cpp

ifdef MODULE_FLOATBV
  CXXFLAGS += -DHAVE_FLOATBV
endif

clean:
	-rm -f $(OBJS) ieee_float_test $(MAINOBJ)
	-rm -f $(OBJDIR)/.deps
#
#ieee_float_test: ieee_float_test.o ieee_float.o util.o
#	$(CXX) $(CXXFLAGS) -o ieee_float_test util.o ieee_float_test.o ../big-int/bigint.o

include $(ESBMCDIR)/common
