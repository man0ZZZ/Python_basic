# -*- coding: utf-8 -*-
"""RockPaperScissor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eymoUMINwEbpUx7ABCOy7B9jM3BaSW6P
"""

###rock scissor paper
# between computer and user
# rock, paper , sci....
# if input varies ask user again for correct input
# game goes on until user wins
# scior < rock
# rock < paper
# paper < sciorz=0
while z<1:
  import random
  input_lst=[0,1,2]
  comp_option=['rock','paper','scissor']
  comp_num=random.choice(input_lst)
  comp=comp_option[(comp_num)]
  user=input("What you want to choose ")
  option=['rock','paper','scissor']
  for i in option:
    if user=='rock' or user=='paper' or user=='scissor':
      continue
    else:
      user=input("Enter correct input 'rock' or 'paper' or 'scissor'")
  print('user', user)
  print('comp', comp)

  if user=='rock' and comp=='scissor':
    z=1
    print("WIN")
  elif user=='paper' and comp=='rock':
    z=1
    print("WIN")
  elif user=='scissor' and comp=='paper':
    z=1
    print("WIN")
  else:
    z=0
    print("TRY AGAIN")