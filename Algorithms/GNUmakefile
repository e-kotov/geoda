
include ../GeoDamake.opt

#include ./file.lst

CPPFLAGS 	:=	$(CPPFLAGS)
CXXFLAGS 	:=	$(CXXFLAGS)

C_SRCS := $(wildcard *.c)
CXX_SRCS := $(wildcard *.cpp)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
MM_SRCS := metal_lisa.mm
else
MM_SRCS :=
endif

OBJ := ${CXX_SRCS:.cpp=.o} ${C_SRCS:.c=.o} ${MM_SRCS:.mm=.o}

default: $(O_OBJ:.o=.$(OBJ_EXT))

clean:
	rm -f *.o 

