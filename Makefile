
CC=g++
CFLAGS+= -O3 -g3 `pkg-config --cflags opencv` -fstack-check -ffast-math -fopenmp
LIBS=`pkg-config --libs opencv`
LIBS+= -lm -lpthread -fopenmp -lgmp -lgmpxx

SOURCES=functions.cpp detect_pedestrian.cpp train_pedestrian.cpp view_values.cpp
OBJECTS=$(SOURCES_DETECT:.cpp=.o)

all: clean detect_pedestrian train_pedestrian view_values
	
%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

detect_pedestrian: functions.o detect_pedestrian.o
	$(CC) $? $(LIBS) -o $@

train_pedestrian: functions.o train_pedestrian.o
	$(CC) $? $(LIBS) -o $@

view_values: functions.o view_values.o
	$(CC) $? $(LIBS) -o $@

clean:
	rm *.o || true
