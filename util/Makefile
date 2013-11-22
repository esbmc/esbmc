all: realall
ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=util.o

SRCS= arith_tools.cpp base_type.cpp cmdline.cpp config.cpp context.cpp \
      expr_util.cpp i2string.cpp language.cpp \
      location.cpp message.cpp language_file.cpp \
      mp_arith.cpp namespace.cpp parseoptions.cpp rename.cpp \
      threeval.cpp typecheck.cpp \
      strstream2string.cpp \
      bitvector.cpp parser.cpp replace_symbol.cpp \
      string_hash.cpp string_container.cpp \
      options.cpp c_misc.cpp simplify_expr.cpp dstring.cpp simplify_expr2.cpp\
      ui_message.cpp simplify_utils.cpp \
      string2array.cpp time_stopping.cpp symbol.cpp \
      type_eq.cpp guard.cpp array_name.cpp message_stream.cpp \
      union_find.cpp \
      xml.cpp xml_irep.cpp std_types.cpp std_code.cpp type_byte_size.cpp \
      format_constant.cpp \
      irep_serialization.cpp symbol_serialization.cpp fixedbv.cpp \
      pointer_arithmetic.cpp signal_catcher.cpp \
      irep2.cpp migrate.cpp crypto_hash.cpp thread.cpp

ifdef MODULE_FLOATBV
  CXXFLAGS += -DHAVE_FLOATBV
endif

clean:
	-rm -f $(OBJS) $(MAINOBJ) irep2.h.gch
	-rm -f $(OBJDIR)/.deps

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
