CC = nvcc
CFLAGS= -Wall -g -O3 
PRGS = contrast
OBJS = contrast-enhancement.cu histogram-equalization.cu  contrast.cu
LIBS = -lm 
CONV = convert
IMGFILE = highres.jpg
INPGM = in.pgm
INPPM = in.ppm
OUTPGM = out.pgm
OUTFILE = outfile.jpg

all:  $(PRGS)

$(PRGS): % :  %.cu
	$(CC) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)
	$(CONV) $(IMGFILE) $(INPGM)
	$(CONV) $(IMGFILE) $(INPPM)

clean:
	rm -rf  *.o  *~ $(PRGS)
	rm -rf $(INPGM)
	rm -rf $(INPPM)
	rm -rf $(OUTPGM)
	rm -rf out_hsl.ppm
	rm -rf out_yuv.ppm
	rm -rf out_blankwhite.jpg
	rm -rf out_high_contrast1.jpg
	rm -rf out_high_contrast2.jpg

run:
	./contrast
	$(CONV) out.pgm out_blankwhite.jpg
	$(CONV) out_hsl.ppm out_high_contrast1.jpg
	$(CONV) out_yuv.ppm out_high_contrast2.jpg
