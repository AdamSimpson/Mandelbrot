#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>
#include "tiffio.h"

#define WORKTAG 1
#define DIETAG 2

struct fracData
{
    int startRow;
    int nRows;
    int nRowsGPU;
    unsigned char *pixels;
};

struct fracInfo
{
    int nCols;
    int nRows;
    double xStart;
    double yStart;
    double radius;
    double spacing;
    int AA;
    double *AAweight;
};

void calcAAweight(struct fracInfo *info)
{
    double norm = 0.0;
    double g;
    int i, j;
    double s = 0.3;
    double x,y;
    int AA = info->AA;

    for(i=0; i<AA; i++) {
        for(j=0; j<AA; j++) {
            x = -0.4 + j*0.8/(AA-1);
            y = -0.4 + i*0.8/(AA-1);
            g = exp(-0.5*(x*x/(s*s)+y*y/(s*s)));
            norm += g;
            info->AAweight[j+i*AA] = g;
        }
    }
    for(i=0; i<AA; i++) {
        for(j=0; j<AA; j++) {
            info->AAweight[j+i*AA] /= norm; 
        }
    }
}

unsigned char MSetPixel(const struct fracInfo *info, double cx, double cy)
{

    static const int maxIter = 2000;
    static const double binBailout = 2000;
    static const double huge = 1e120;
    static const double overflow = 1e300;

    double spacing = info->spacing;
    double radius = info->radius;
    int AA = info->AA;

    double pixVal = 0.0;

    int subCx, subCy;
    double subX, subY;
    for(subCy=0; subCy<AA; subCy++) {
        for(subCx=0; subCx<AA; subCx++) {
            subY = (cy - spacing/2.0) + subCy*spacing/AA;
            subX = (cx - spacing/2.0) + subCx*spacing/AA;
        
            //We can also look at the reciprical complex plane
            double tmpX = subX;
            subX = subX/(subX*subX+subY*subY);
            subY = -1.0*subY/(tmpX*tmpX+subY*subY);
        
            double x = 0 ,y = 0;
            double x2 = 0, y2 = 0;
            double dist = 0;
            double xOrbit = 0, yOrbit = 0;
            int iter = 0;
            double tmp;
            bool flag = false;
            double xder = 0, yder = 0;
            double yBailout = 0;
            bool binBailed = false;
        
            while(iter < maxIter)
            {
                if(flag)
                    break;

                tmp = x2-y2+subX;
                y = 2*x*y+subY;
                x = tmp;
                x2 = x*x;
                y2 = y*y;
                iter++;
            
                tmp = 2*(xOrbit*xder-yOrbit*yder)+1;
                yder = 2*(yOrbit*xder + xOrbit*yder);
                xder = tmp;
                flag = fmax(fabs(xder), fabs(yder)) > overflow;
            
                //If too large of a bailout is used the binary looks bad
                //This should collect the first y after we reach binBailout
                if(x2 + y2 > binBailout && !binBailed) {
                    yBailout = y;
                    binBailed =  true;
                }
            
                if (x2 + y2 > huge) {
                    dist = log(x2+y2)*sqrt(x2+y2)/sqrt(xder*xder+yder*yder);
                    break;
                }
            
                xOrbit = x;
                yOrbit = y;
            }
        
            // Distance estimator coloring
            if(dist <= radius)
                pixVal += pow(dist/radius, 1.0/3.0)*info->AAweight[subCx+subCy*AA];
	    // "Padding" between binary and distance
            else if(iter > 30)
                pixVal += 1.0*info->AAweight[subCx+subCy*AA];
	    // Binary black
            else if(yBailout > 0)
                pixVal += 0.0;
	    // Binary white
            else
                pixVal += 1.0*info->AAweight[subCx+subCy*AA];
        
        }//subCx
    }//SubCy

    //Implicit cast to char
    return round(pixVal*255);

}

void get_work(const struct fracInfo *info, int *rowsTaken, struct fracData *work)
{
    if(*rowsTaken >= info->nRows){
        work->nRows = 0;
        return;
    }
    int rows = 16;

    work->startRow = *rowsTaken;
    int numRows = (*rowsTaken)+rows<info->nRows?rows:info->nRows-(*rowsTaken);
    work->nRows = numRows;

    *rowsTaken += numRows;
}

int get_max_work_size(const struct fracInfo *info)
{
    return 16*info->nCols;
}

void calcPixels(const struct fracInfo *info, struct fracData *data)
{
    int nx = info->nCols;
    int ny = data->nRows;
    double spacing = info->spacing;
    double xStart = info->xStart;
    double yStart = info->yStart + (data->startRow*spacing);
    int ix,iy;
    double cx, cy;

    #pragma omp parallel for private(ix,cx,iy,cy) shared(info,nx,ny,xStart,yStart) schedule(dynamic) 
    for(iy=0; iy<ny; iy++) {
        cy = yStart + iy*spacing;
        for(ix=0; ix<nx; ix++) {
            cx = xStart + ix*spacing;
            data->pixels[iy*nx+ix] = MSetPixel(info, cx, cy);
        }
    }
}

void master(const struct fracInfo info)
{
    int ntasks, dest, msgsize;
    struct fracData *work = malloc(sizeof(*work));
    MPI_Status status;
    int rowsTaken = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);    

    size_t size = sizeof(unsigned char) * (unsigned long)info.nCols * (unsigned long)info.nRows;
    unsigned char *fractal = (unsigned char*)malloc(size);
    if(!fractal) {
        printf("fractal allocation failed, %lu bytes\n", size);
        exit(1);
    }

    // Allocate buffer
    int membersize, emptysize, fullsize;
    int position;
    char *buffer;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &membersize);
    emptysize = membersize;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &membersize);
    emptysize += membersize;
    MPI_Pack_size(get_max_work_size(&info), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD, &membersize);
    fullsize = emptysize + membersize;

    buffer = malloc(fullsize);    
    if(!buffer) {
        printf("buffer allocation failed, %d bytes\n",fullsize);
        exit(1);
    }

    // Send initial data
    for (dest = 1; dest < ntasks; dest++) {
        //Get next work item
        get_work(&info,&rowsTaken,work);
        
        //pack and send work       
        position = 0;
        MPI_Pack(&work->startRow,1,MPI_INT,buffer,emptysize,&position,MPI_COMM_WORLD);
        MPI_Pack(&work->nRows,1,MPI_INT,buffer,emptysize,&position,MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, dest, WORKTAG, MPI_COMM_WORLD);
    }

    printf("sent initial work\n");
    //Get next work item
    get_work(&info,&rowsTaken,work);
    int startRow, nRows;
    while(work->nRows) {
        // Recieve and unpack work
        MPI_Recv(buffer, fullsize, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        position = 0;
        MPI_Get_count(&status, MPI_PACKED, &msgsize);
        MPI_Unpack(buffer, msgsize, &position, &startRow,1,MPI_INT,MPI_COMM_WORLD);
        MPI_Unpack(buffer, msgsize, &position, &nRows,1,MPI_INT,MPI_COMM_WORLD);    
        MPI_Unpack(buffer, msgsize, &position, fractal+((unsigned long)startRow*info.nCols), nRows*info.nCols, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

        //pack and send work       
        position = 0;
        MPI_Pack(&work->startRow,1,MPI_INT,buffer,emptysize,&position,MPI_COMM_WORLD);
        MPI_Pack(&work->nRows,1,MPI_INT,buffer,emptysize,&position,MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, status.MPI_SOURCE, WORKTAG, MPI_COMM_WORLD);

        //Get next work item
        get_work(&info,&rowsTaken,work);

        if(status.MPI_SOURCE==1)
            printf("%d\n",work->startRow);
    }

    // Recieve all remaining work
    for (dest = 1; dest < ntasks; dest++) {
        // Recieve and unpack work
        MPI_Recv(buffer, fullsize, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        position = 0;
        MPI_Get_count(&status, MPI_PACKED, &msgsize);

        MPI_Unpack(buffer, msgsize, &position, &startRow,1,MPI_INT,MPI_COMM_WORLD);
        MPI_Unpack(buffer, msgsize, &position, &nRows,1,MPI_INT,MPI_COMM_WORLD);
        // unpack pixel data
        MPI_Unpack(buffer, msgsize, &position, fractal+((unsigned long)startRow*info.nCols), nRows*info.nCols, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

        // Kill slaves
        MPI_Send(0,0,MPI_INT,dest,DIETAG,MPI_COMM_WORLD);
    }

    free(work);
    free(buffer);

    //Save image as TIFF
    unsigned int nx = info.nCols;
    unsigned int ny = info.nRows;
    char fileName[] = "/home/pi/Mandelbrot/Mandelbrot.tiff";
    TIFF *out = TIFFOpen(fileName, "w");
    uint32 tileDim = 256;
    tsize_t tileBytes = tileDim*tileDim*sizeof(char);
    unsigned char *buf = (unsigned char *)_TIFFmalloc(tileBytes);
    char description[1024];
    snprintf(description, sizeof(description),"xStart:%f yStart:%f spacing:%f AAx:%d",info.xStart,info.yStart,info.spacing,info.AA);
    TIFFSetField(out, TIFFTAG_IMAGEDESCRIPTION, description);
    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, (uint32) nx);
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, (uint32) ny);
    TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
    TIFFSetField(out, TIFFTAG_TILEWIDTH, tileDim);
    TIFFSetField(out, TIFFTAG_TILELENGTH,  tileDim);
//    TIFFSetField(out, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
//    TIFFSetField(out, TIFFTAG_XRESOLUTION, resolution);
//    TIFFSetField(out, TIFFTAG_YRESOLUTION, resolution);
//    TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);    
    unsigned long x,y,i,j;
    unsigned long tileStart;
    // Iterate through and write tiles
    for(y=0; y<ny; y+=tileDim) {
        for(x=0; x<nx; x+=tileDim) {
            // Fill tile with fractal data
            tileStart = y*nx+x;
            for(i=0; i<tileDim; i++) {
                for(j=0; j<tileDim; j++) {
                    if(x+j < nx && y+i < ny)
                        buf[i*tileDim+j] = fractal[(y+i)*nx+(x+j)];
                    else
                        buf[i*tileDim+j] = (unsigned char)0;
                }
            }
            TIFFWriteTile(out, buf, x, y, 0, 0);
        }
    }
    
    TIFFClose(out);
    _TIFFfree(buf);
    free(fractal);
}

void slave(const struct fracInfo info)
{
    MPI_Status status;
    int msgsize;
    struct fracData *data = malloc(sizeof(*data));
    data->pixels = (unsigned char*)malloc(get_max_work_size(&info)*sizeof(unsigned char));  

    // Allocate buffers
    int membersize, emptysize, fullsize;
    int position;
    char *buffer; //Contains no pixel data
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &membersize);
    emptysize = membersize;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &membersize);
    emptysize += membersize;
    MPI_Pack_size(get_max_work_size(&info), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD, &membersize);
    fullsize = emptysize+membersize;
    buffer = malloc(fullsize);

    while(1) {
        // Recieve and unpack work
        MPI_Recv(buffer, emptysize, MPI_PACKED, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // Check tag for work/die
        if(status.MPI_TAG == DIETAG) {
            return;
        }

        // Unpack work info
        position = 0;
        MPI_Get_count(&status, MPI_PACKED, &msgsize);
        MPI_Unpack(buffer, msgsize, &position, &data->startRow,1,MPI_INT,MPI_COMM_WORLD);
        MPI_Unpack(buffer, msgsize, &position, &data->nRows,1,MPI_INT,MPI_COMM_WORLD);

        // calcPixels
        calcPixels(&info, data);        

        // Pack and send data back
        position = 0;
        MPI_Pack(&data->startRow,1,MPI_INT,buffer,fullsize,&position,MPI_COMM_WORLD);
        MPI_Pack(&data->nRows,1,MPI_INT,buffer,fullsize,&position,MPI_COMM_WORLD);
        MPI_Pack(data->pixels, data->nRows*info.nCols, MPI_UNSIGNED_CHAR,buffer,fullsize,&position,MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, 0, WORKTAG, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    struct fracInfo info;
    //Dimensions of grid
    const double xMin = -1.8;
    const double xMax = 4.5;
    const double yMin = -1.8;
    const double yMax = 1.8;

    //number of pixels in x and y
    const int nx = 600;
    const int ny = ceil(nx*(yMax-yMin)/(xMax-xMin));
    const double spacing = (xMax-xMin)/(nx-1);

    info.AA = 2; //AntiAliasing is NxN
    info.AAweight = malloc(info.AA*info.AA*sizeof(double));
    const double threshold = 1.0;

    // Set frac info struct
    info.nCols = nx;
    info.nRows = ny;
    info.xStart = xMin;
    info.yStart = yMin;
    info.radius = threshold * 0.5*spacing/info.AA;
    info.spacing = spacing;

    calcAAweight(&info);

    // Initialize MPI
    int myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0){
        printf("Resolution: %dx%d\n", nx, ny);
        master(info);
    }
    else
        slave(info);

    MPI_Finalize();

    return 0;
}
