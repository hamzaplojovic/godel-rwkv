CC      = cc
CFLAGS  = -O2 -std=c99 -Wall -Wextra -Wpedantic -lm
TARGET  = ../bin/godelrd
SRCS    = daemon.c godel.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) godel.h
	@mkdir -p ../bin
	$(CC) $(CFLAGS) -o $@ $(SRCS)
	@echo "Built: $@"

clean:
	rm -f $(TARGET)
