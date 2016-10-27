#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <functional>

struct simple_interop_t {
  const unsigned int width, height;
  GLuint mtexture;

  uchar3* h_textureBufferData = nullptr;
  uchar3* d_textureBufferData = nullptr;

  GLuint gl_pixelBufferObject = 0;
  cudaGraphicsResource * cudaPboResource = nullptr;

  simple_interop_t(unsigned int w, unsigned int h) : width(w), height(h) {
    glEnable(GL_TEXTURE_2D);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glGenTextures(1, &mtexture);
    glBindTexture(GL_TEXTURE_2D, mtexture);

    glTexImage2D(GL_TEXTURE_2D,
                 0,                    // level 0
                 3,                    // use only R, G, and B components
                 width, height,        // texture has width x height texels
                 0,                    // no border
                 GL_RGB,               // texels are in RGB format
                 GL_UNSIGNED_BYTE,     // color components are unsigned bytes
                 h_textureBufferData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenBuffers(1, &gl_pixelBufferObject);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar3),
                 h_textureBufferData, GL_STREAM_COPY);
 
    cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject,
                                                    cudaGraphicsMapFlagsWriteDiscard);

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

  }

  void cuda_render(std::function<void(uchar3*)> ff) {
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData,
                                         &num_bytes, cudaPboResource);
    ff(d_textureBufferData);
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

    draw_quad();
  }

  void draw_quad() {
    glClear(GL_COLOR_BUFFER_BIT);  
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, mtexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
 
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    width,height,
                    GL_RGB, GL_UNSIGNED_BYTE, 0);
  
    float x = 1.0;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-x, -x, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, -x, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(x, x, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-x, x, 0.5);
    glEnd();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  
};

