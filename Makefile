CXX=g++-4.9

HGVERSION:= $(shell hg parents --template '{node|short}')

CXXFLAGS_COMMON = \
	-Wall -Wextra -std=gnu++11 -mtune=native -Wa,-q -ffast-math \
	-pthread -fopenmp -DHGVERSION="\"${HGVERSION}\""
	
CXXFLAGS_DEP = \
	-std=gnu++11

CXXFLAGS_RELEASE = $(CXXFLAGS_COMMON) -march=native -O3 -flto
CXXFLAGS_DEBUG = $(CXXFLAGS_COMMON) -g

ifeq ($(DEBUG),1)
	CXXFLAGS=$(CXXFLAGS_DEBUG)
else
	CXXFLAGS=$(CXXFLAGS_RELEASE)
endif

CXXFILES := \
	$(wildcard *.cpp) \
	$(wildcard ann/*.cpp) \
	$(wildcard eval/*.cpp)

INCLUDES=-I. -IEigen_dev

EXE=giraffe

LDFLAGS=-L. -lm -ltcmalloc

OBJS := $(CXXFILES:%.cpp=obj/%.o)
DEPS := $(CXXFILES:%.cpp=dep/%.d)

ifeq ($(V),0)
	Q = @
else
	Q =
endif

ifeq ($(OS),Windows_NT)
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	endif
	ifeq ($(UNAME_S),Darwin)
		# OSX needs workaround for AVX, and LTO is broken
		# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47785
		CXXFLAGS += -Wa,-q
		CXXFLAGS := $(filter-out -flto,$(CXXFLAGS))
    endif
endif

.PHONY: clean test

default: $(EXE)

dep/%.d: %.cpp
	$(Q) $(CXX) $(CXXFLAGS_DEP) $(INCLUDES) $< -MM -MT $(@:dep/%.d=obj/%.o) > $@
	
obj/%.o :
	$(Q) $(CXX) $(CXXFLAGS) $(INCLUDES) -c $(@:obj/%.o=%.cpp) -o $@

$(EXE): $(OBJS)
	$(Q) $(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) -o $(EXE)

test:
	$(Q) echo $(DEPS)
	
clean:
	-$(Q) rm -f $(DEPS) $(OBJS) $(EXE)

ifneq ($(MAKECMDGOALS),clean)
    -include $(DEPS)
endif
