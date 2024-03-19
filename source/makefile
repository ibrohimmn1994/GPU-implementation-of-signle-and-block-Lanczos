


suf=$(suffix $(target))

ifeq ($(suf),.cu)
CC = nvcc
LIBS = -lcublas -lcusolver 

ARCH = -arch=sm_75
CFLAGS =
EXT = cuda 

endif
ifeq ($(suf),.cpp)
CC = g++
LIBS = 
ARCH = 
CFLAGS = -O3
EXT = host
endif


BIN = $(addsuffix .$(EXT),$(basename $(target)))
all: $(BIN)
	@echo " execution file -> ${BIN}"	
$(BIN):  $(target)
	$(CC) $(CFLAGS) $(ARCH) -o $@ $< $(LIBS) 
clean:
	$(RM) $(BIN) 
	
.PHONY: clean
		

