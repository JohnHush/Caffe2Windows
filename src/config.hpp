#ifndef __JohnHush_CONFIG_H
#define __JohnHush_CONFIG_H

// define the header to build .dll in WINDOWS
#ifdef _WINDOWS
    #define OCRAPI __declspec(dllexport)
#else
    #define OCRAPI
#endif


// system file operation function include file
#ifdef _WINDOWS
#include <windows.h>
#endif

#ifdef APPLE
#include <mach-o/dyld.h>
#endif

#ifdef UNIX
#include <unistd.h>
#endif

#endif
