all:	mandelbrotAA mandelbrot

.PHONY:	clean

mandelbrotAA:	mandel-DEM-BD-AA-hybrid.c
	mpicc -lm -ltiff mandel-DEM-BD-AA-hybrid.c -o mandelbrotAA
mandelbrot:	mandel-DEM-BD.c
	mpicc -lm -ltiff mandel-DEM-BD.c -o mandelbrot

clean:
	rm -rf *.o
