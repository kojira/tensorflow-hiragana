#!/usr/bin/env python
# -*- coding: utf-8 -*-

shift_jis = []
jisx0208 = []
unicode = []
with open("JIS0208.TXT", "r") as f:
    for line in f:
        if line[0] == "#":
            pass
        else:
            sjis, jisx, unic, _ = line.strip().split("\t")
            shift_jis.append(int(sjis,16))
            jisx0208.append( int(jisx,16))
            unicode.append(  int(unic,16))

def jisx0208_2uni(n):
    return unicode[jisx0208.index(n)]
    
def uni2jisx0208(n):
    return jisx0208[unicode.index(n)]

def uni2sjis(n):
    return shift_jis[unicode.index(n)]

def sjis2jisx0208(n):
    return jisx0208[shift_jis.index(n)]

if __name__ == '__main__':
    print("{:x}".format(jisx0208_2uni(0x2422)))
    print(chr(jisx0208_2uni(0x2422)))
    print(str(uni2jisx0208(0x3042)))
    print(chr(jisx0208_2uni(uni2jisx0208(0x3042))))
    print(ord('A'))
    print(ord('a'))
    print("{:x}".format((uni2jisx0208(ord('ヲ')))))
    print("{:x}".format((uni2sjis(ord('ヲ')))))
    print("{:x}".format((sjis2jisx0208(uni2sjis(ord('ヲ'))))))
    print("{:x}".format((ord('ｦ'))))
    print("{:x}".format((ord('ｱ'))))
    print("{:x}".format((ord('ｲ'))))
