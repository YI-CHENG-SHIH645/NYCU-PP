CXX = g++
CFLAGS = -g -Wall -O3 --std=c++17 -lpthread
INCLUDES = -I FastRand -I pcg-cpp-0.98/include

pi.out:
	$(CXX) $(CFLAGS) $(INCLUDES) pi.cpp -o pi.out

clean:
	rm pi.out
