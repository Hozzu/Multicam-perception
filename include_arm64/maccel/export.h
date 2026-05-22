
#ifndef MACCEL_EXPORT_H
#define MACCEL_EXPORT_H

#ifdef MACCEL_STATIC_DEFINE
#  define MACCEL_EXPORT
#  define MACCEL_NO_EXPORT
#else
#  ifndef MACCEL_EXPORT
#    ifdef maccel_EXPORTS
        /* We are building this library */
#      define MACCEL_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define MACCEL_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef MACCEL_NO_EXPORT
#    define MACCEL_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef MACCEL_DEPRECATED
#  define MACCEL_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef MACCEL_DEPRECATED_EXPORT
#  define MACCEL_DEPRECATED_EXPORT MACCEL_EXPORT MACCEL_DEPRECATED
#endif

#ifndef MACCEL_DEPRECATED_NO_EXPORT
#  define MACCEL_DEPRECATED_NO_EXPORT MACCEL_NO_EXPORT MACCEL_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef MACCEL_NO_DEPRECATED
#    define MACCEL_NO_DEPRECATED
#  endif
#endif

#endif /* MACCEL_EXPORT_H */
