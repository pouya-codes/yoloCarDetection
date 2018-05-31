#-------------------------------------------------
#
# Project created by QtCreator 2018-05-30T13:29:13
#
#-------------------------------------------------

QT       += core gui
CONFIG += c++11



greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = occupantDetection
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv
