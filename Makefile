CXX = g++
CXXFLAGS = -Wall -std=c++17 -O2
LIBS = -lsfml-graphics -lsfml-window -lsfml-system

TARGET = EvoGAQLearningSim
SOURCES = EvoGAQLearningSim.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
    $(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LIBS)

%.o: %.cpp
    $(CXX) $(CXXFLAGS) -c $<

clean:
    rm -f $(OBJECTS) $(TARGET)