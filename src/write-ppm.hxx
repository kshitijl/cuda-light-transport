#pragma once

#include <stdio.h>

void write_ppm(const char * filename, int width, int height, char * data) {
  FILE *fp = fopen(filename, "wb");
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  fwrite(data, width*height*3, 1, fp);
  fclose(fp);
}
