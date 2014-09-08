TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    giraffe.cpp \
    magic_moves.cpp \
    board_consts.cpp \
    board.cpp \
    eval/eval.cpp \
    search.cpp \
    see.cpp \
    backend.cpp \
    chessclock.cpp \
    timeallocator.cpp \
    zobrist.cpp \
    ttable.cpp

HEADERS += \
    board_consts.h \
    magic_moves.h \
    types.h \
    board.h \
    move.h \
    bit_ops.h \
    containers.h \
    util.h \
    eval/eval.h \
    search.h \
    see.h \
    backend.h \
    chessclock.h \
    timeallocator.h \
    eval/eval_params.h \
    zobrist.h \
    ttable.h

