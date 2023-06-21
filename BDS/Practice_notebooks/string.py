# -*- coding: utf-8 -*-
"""String

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19XhPPTEmnKYYXKVTWxyCMFX4O8i4daYS
"""

#positive indexing and positive splicing
str1 = 'Harry Potter - Goblet of Fire'
print(str1[::])
print(str1[:5:])#element 1 and 3 is empty, so takes the defalut value of 1st element is 0 and 3rd element is 1
print(str1[6:12:])
print(str1[15:21:])
print(str1[22:24:])
print(str1[25::])#second element is 0, its default value is length of string, i.e.29

#negative indexing
#third element is necessary to go from right to left ,i.e. -1
print(str1[::-1])
print(str1[25::-1])
print(str1[-1:-5:-1])
print(str1[:-5:-1])#pick up last 5 indexes
#negative indexing and positive slicing
print(str1[-4::])

print(str1[:-5:-1])
print(str1[-6:-8:-1])
print(str1[-9:-15:-1])
print(str1[-18:-24:-1])
print(str1[-25::-1])

str1='Harry Potter - Goblet of Fire' # first think about direcrion, forward or reverse, then think of index, then the start and end,
#+ve indexing +ve slicing(left to right)
print(str1[0:len(str1):1])
print(-(len(str1)+1))

#-ve indexing and -ve slicing(right to left)
print(str1[-1:-(len(str1)+1):-1])

#postitive indexing(first two element should be positive) and negative slicing
print(str1[28::-1])

#-ve indexing and +ve slicing
print(str1[-29::])

str1="PYTHON PROGRAMMING"
print(-(len(str1)+1))
#+ve indexing +ve slicing
print(str1[:len(str1):])
#-ve indexing -ve slicing
print(str1[::-1])
print(str1[:(-(len(str1)+1)):-1])
#+ve indexing and -ve slicing
print(str1[18::-1])
#-ve indexing and +ve slicing
print(str1[-18::])

str1='Ganga Brahmaputra Narmada Godaveri Kaveri Yamuna'
print(len(str1))
print(-(len(str1)+1))
#+ve indexing +ve slicing
print(str1[::])
print(str1[:len(str1):])
#-ve indexing -ve slicing
print(str1[::-1])
print(str1[-1:(-(len(str1)+1)):-1])
#+ve indexing and -ve slicing
print(str1[48::-1])
#-ve indexing and +ve slicing
print(str1[-48::])

#list
lst1=["Ganga", "Brahmaputra", "Narmada", "Godaveri", "Kaveri", "Yamuna"]
print(lst1)
print(lst1[:1:], lst1[])

lst1=[1,['a','b','c'],1.2,2,2+3j,False,'apple',['mon','tue','wed'],3,3.4,4,True,-4-5j,'banana',5,5.6,True,6,-6+7j,'cherry',False]
print(len(lst1))
print(-(len(lst1)+1))
#+ve indexing +ve slicing
print(lst1[::])
print(lst1[:len(lst1):])
#-ve indexing -ve slicing
print(lst1[::-1])
print(lst1[-1:(-(len(lst1)+1)):-1])
#+ve indexing and -ve slicing
print(lst1[21::-1])
#-ve indexing and +ve slicing
print(lst1[-21::])