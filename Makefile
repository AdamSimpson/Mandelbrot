all:	mandelbrot

.PHONY:	clean

mandelbrot:	mandel-DEM-BD-AA-hybrid.c
	mpicc -lm -ltiff mandel-DEM-BD-AA-hybrid.c -o mandelbrot

clean:
	rm -rf *.o
