INCLUDE = -Isysport/ \
	-Icompression/ \
	-Icompression/liblzf/ \
	-Icompression/zlib/ \
	-Icompression/lzma/ \
	-Icompression/huffman/

DEFAULT_DEFINE = -DZ_PREFIX
DEFAULT_CFLAGS = -Wall -Wextra $(INCLUDE) $(DEFAULT_DEFINE) -static $(CFLAGS)
OPTFLAGS = -O3 -march=native -DNDEBUG

SRCFILES := gtb-probe.c gtb-dec.c gtb-att.c sysport/sysport.c \
	compression/wrap.c compression/huffman/hzip.c \
	compression/lzma/LzmaEnc.c compression/lzma/LzmaDec.c \
	compression/lzma/Alloc.c compression/lzma/LzFind.c \
	compression/lzma/Lzma86Enc.c compression/lzma/Lzma86Dec.c \
	compression/lzma/Bra86.c compression/zlib/zcompress.c \
	compression/zlib/uncompr.c compression/zlib/inflate.c \
	compression/zlib/deflate.c compression/zlib/adler32.c \
	compression/zlib/crc32.c compression/zlib/infback.c \
	compression/zlib/inffast.c compression/zlib/inftrees.c \
	compression/zlib/trees.c compression/zlib/zutil.c \
	compression/liblzf/lzf_c.c compression/liblzf/lzf_d.c
OBJFILES := $(patsubst %.c,%.o,$(SRCFILES))

LIBNAME := libgtb.a

ifeq ($(OS),Windows_NT)
	CC=gcc
endif

.PHONY: all clean
.DEFAULT_GOAL := all

all:
	$(MAKE) $(LIBNAME) \
		ARCHFLAGS='$(DEFAULT_ARCHFLAGS)' \
		DEFINE='$(DEFAULT_DEFINE)' \
		CFLAGS='$(OPPTFLAGS) $(DEFAULT_CFLAGS)'

$(LIBNAME): $(OBJFILES)
	$(AR) rcs $@ $(OBJFILES)

clean:
	$(RM) -f $(OBJFILES) $(LIBNAME)

.depend:
	$(CC) -MM $(DEFAULT_CFLAGS) $(SRCFILES) > $@
	
windows_clean:
	del $(subst /,\,$(OBJFILES)) $(LIBNAME)

include .depend

