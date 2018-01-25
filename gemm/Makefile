# Makefile

MKLPATH=$(MKLROOT)/lib/intel64
MKLINCLUDE=$(MKLROOT)/include

PROGRAM = main
OBJ     = main.o
CC      = icc
CFLAGS  = -Wall -O3 -std=c99
HDR_DIR = -I$(MKLINCLUDE)
LIB_DIR = -L$(MKLPATH)
LIB     = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lifcore
LIBLINK = $(HDR_DIR) $(LIB_DIR) $(LIB)

.PHONY: all
all: $(PROGRAM)

.PHONY: $(PROGRAM)
$(PROGRAM): $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIBLINK)

.c.o: $(TESTER)
	$(CC) $(CFLAGS) -c $< $(HDR_DIR)

.PHONY: clean
clean:
	rm $(PROGRAM) $(OBJ)
