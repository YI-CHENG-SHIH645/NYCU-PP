scp ~/Documents/NYCU/PP/HW1/part1/def.h PP_server:HW1/part1/;
ssh PP_server 'cd HW1/part1; make clean; make; ./myexp -s 10000; make clean;'
