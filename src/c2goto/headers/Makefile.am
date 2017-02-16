noinst_LTLIBRARIES = libheaders.la
libheaders_la_SOURCES = digitalfilter.c pthread.c pthreadtypes.c stdarg.c \
                       stdbool.c stddef.c limits.c
SUFFIXES = .hs

.hs.c:
	$(srcdir)/flail.sh $< $< > $@

clean-local:
	-rm -rf *.c
