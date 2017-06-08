noinst_LTLIBRARIES = libutil.la
libutil_la_SOURCES = \
      irep2.cpp \
      arith_tools.cpp base_type.cpp cmdline.cpp config.cpp context.cpp \
      expr_util.cpp i2string.cpp language.cpp location.cpp message.cpp \
      language_file.cpp mp_arith.cpp namespace.cpp parseoptions.cpp rename.cpp \
      threeval.cpp typecheck.cpp bitvector.cpp parser.cpp replace_symbol.cpp \
      string_hash.cpp string_container.cpp options.cpp c_misc.cpp \
      simplify_expr.cpp dstring.cpp simplify_expr2.cpp ui_message.cpp \
      simplify_utils.cpp string2array.cpp time_stopping.cpp symbol.cpp \
      type_eq.cpp guard.cpp array_name.cpp message_stream.cpp union_find.cpp \
      xml.cpp xml_irep.cpp std_types.cpp std_code.cpp format_constant.cpp \
      irep_serialization.cpp symbol_serialization.cpp fixedbv.cpp \
      signal_catcher.cpp migrate.cpp \
      thread.cpp crypto_hash.cpp type_byte_size.cpp dcutil.cpp \
      string_constant.cpp c_types.cpp ieee_float.cpp c_qualifiers.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -Wno-bool-compare

utilincludedir = $(includedir)/util
utilinclude_HEADERS = arith_tools.h array_name.h base_type.h bitvector.h \
      bp_converter.h c_misc.h c_types.h cmdline.h \
      config.h context.h cprover_prefix.h crypto_hash.h dcutil.h \
      dstring.h expr.h expr_util.h fixedbv.h \
      format_constant.h format_spec.h global.h guard.h hash_cont.h \
      i2string.h ieee_float.h irep.h irep2.h irep_serialization.h \
      language.h language_file.h location.h message.h message_stream.h \
      migrate.h mp_arith.h namespace.h numbering.h options.h \
      parseoptions.h parser.h prefix.h c_qualifiers.h \
      reference_counting.h rename.h replace_symbol.h signal_catcher.h \
      simplify_expr.h simplify_expr_class.h simplify_utils.h std_code.h \
      std_expr.h std_types.h string2array.h string_constant.h \
      string_container.h string_hash.h symbol.h symbol_serialization.h \
      thread.h threeval.h time_stopping.h type.h type_byte_size.h \
      type_eq.h typecheck.h ui_message.h union_find.h xml.h xml_irep.h

