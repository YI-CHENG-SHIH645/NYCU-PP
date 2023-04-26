MPICXX = mpicxx
CXXFLAGS = -O3
EXEC = matmul

$(EXEC): main.o matmul.o
	$(MPICXX) $(CXXFLAGS) $^ -o $@

main.o: main.cc
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

matmul.o: matmul.cc
	$(MPICXX) $(CXXFLAGS) -c $< -o $@
