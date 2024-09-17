# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lfftw3f

# Directories
SRCDIR = .
OBJDIR = obj
BINDIR = bin

# Source and object files
SRC = $(SRCDIR)/FFTWrapper.cpp $(SRCDIR)/convolution.cpp
OBJ = $(OBJDIR)/FFTWrapper.o $(OBJDIR)/convolution.o

# Output binary
TARGET = $(BINDIR)/my_program

# Default target
all: $(TARGET)

# Linking the binary
$(TARGET): $(OBJ)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJ)

# Compiling source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean the build
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean
