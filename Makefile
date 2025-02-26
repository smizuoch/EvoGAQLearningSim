NAME = sim

SRCS = EvoGAQLearningSim.cpp

OBJS = $(SRCS:.cpp=.o)

CXX = g++

CXXFLAGS = -lsfml-graphics -lsfml-window -lsfml-system

all: $(NAME)

$(NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(NAME) $(OBJS)

clean:
	rm -f $(OBJS)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean re