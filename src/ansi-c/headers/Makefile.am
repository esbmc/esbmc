noinst_LIBRARIES = libheaders.a
libheaders_a_SOURCES = digitalfilter.hs pthread.hs pthreadtypes.hs stdarg.hs \
                       stdbool.hs stddef.hs limits.hs
SUFFIXES = .hs

.hs.o:
	$(srcdir)/flail.sh $< $< | $(CC) $(CFLAGS) -x c - -c -o $@
