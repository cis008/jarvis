'''try:
    x=int(input("x is:"))
    print(f"the ans is:{x}")
except ValueError:
    print("x is not an integer ")'''
'''i=1 
while(i<=5):
       print(i,"-",i*i)
       i=i+1'''
'''for i in range(4): 
        print(4*"*")'''
'''temperature=25.5
celsius=temperature
kelvin=celsius+273
print(f"celsius={celsius}")
print(f"kelvin={kelvin}")'''
#question 1
'''a=[1,-2,3,-4,5,6,-7,-8,9,10]
pnc=0
for num in a:
    if num>0:
        pnc+=1
print("The final count is:",pnc)'''
#question 2
'''n=10
sum_even=0
for num in range(1,n+1):
    if n%2==0:
        sum_even+=2
print(sum_even)'''
#question 3
'''number=3
for i in range(1,11):
    if i==5:
        continue
    print(number,"x",i,"=",number*i)'''
#question 4
'''Str="Python"
reversed_ch=""
for char in Str:
    reversed_ch=char+reversed_ch
print(reversed_ch)'''
#question 5
'''Str=input("")
for char in Str:
   if Str.count(char)==1:
       print("char is",char)'''
#question 6
'''number=5
factorial=1
while number>0:
    factorial=factorial*number
    number-=1
print("The factorial is",factorial)'''
#question 7
'''n=int(input("pls enter the number btw 1 and 10: "))
if 1<=n<=10:
    print("Thank you for entering a number")
    
else:
    print("invalid number")'''
#question 8
'''n=int(input("enter a number: "))
is_prime=True
if n>1:
    for i in range(2,n):
        if n%i==0:
            is_prime=False
            break
        
        
print(is_prime)'''
#exercise 2 question 1
'''i=0
while i<11:
    print(i)
    i+=1'''
#exercise 2 question 2
'''r=int(input())
for i in range(1,r+1):
    for j in range(1,i+1):
        print(j,end="")'''
# Decide the row count. (above pattern contains 5 rows)
'''row = 5
# start: 1
# stop: row+1 (range never include stop number in result)
# step: 1
# run loop 5 times
for i in range(1, row + 1):
    # Run inner loop i+1 times
    for j in range(1, i + 1):
        print(j, end=' ')
    # empty line after each row
    print("")'''
'''n=int(input())
for i in range(1,n+1):
    for j in range(1,i+1):
        print(j,end="")
    print("")'''
#exercise 2 question 3
'''n=int(input())
i=1
sum=0
while(i<=n):
    sum=sum+i
    i+=1
print(sum)'''
#exercise 2 question 4
'''n=int(input())
for i in range(1,n+1):
    print("2","X",i,"=",2*i)'''
#exercise 2 question 5
'''numbers=[12,75,150,100,145,525,50]
for number in numbers:
    if number>500:
        break
    elif number>150:
        continue
        
    elif number%5==0:
        print(number)'''
# exercise 2 question 6
'''n=int(input())
x=len(str(n))
print(x)'''
#exercise 2 question 7
'''n=int(input())
for i in range(1,n+1):
    for j in range(1,i+1):
        print(j,end="")
    print()'''
'''import random
x=["heads","tails"]
random.shuffle(x)
for t in x:
    print(t)'''
'''import statistics
print(statistics.mode([200,100,200]))'''

'''import pyttsx3
import speech_recognition as sr
engine = pyttsx3.init()
engine.say("Hello, world!")
engine.runAndWait()'''
'''import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth
client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

scope = "user-read-playback-state,user-modify-playback-state"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='YOUR_CLIENT_ID',
                                               client_secret='YOUR_CLIENT_SECRET',
                                               redirect_uri='http://localhost:8888/callback',
                                               scope=scope))

# Test API call
try:
    current_playback = sp.current_playback()
    print("Successfully connected to Spotify.")
except Exception as e:
    print("Error connecting to Spotify:", e)
'''

'''import numpy as np
import matplotlib as plt
import pandas as pd
ds=pd.read_csv('C://Users//Archit Jagtap//Desktop//last try//Mllearn.py//Data.csv')
x=ds.iloc[:, -1]
y=ds.iloc[:, -1]
print(ds)'''
#learning DSA 
'''stock_prices=[298,320,312,300,295]   
stock_prices.remove(320)
for price in stock_prices:
    print(price)'''
#Array problem set 1 
#question 1 
'''exp=[2200,2350,2600,2130,2190]
print ( exp[1]-exp[0])
print(exp[0]+exp[1]+exp[2])
for price in exp:
    if price==2000:
        print("yes")
exp.append(1980)
print(exp)
exp[3]=exp[3]-200
print(exp)'''
#question 2 
'''heros=['spider man','thor','hulk','iron man','captain america']
heros.append("black panther")
heros.remove("black panther")
heros.insert(3,"black panther")
heros[1:3]=["doctor Strange"]
heros.sort()
print(heros)
length=len(heros)
print(length)'''
#question 3 
'''max=int(input("pls enter the number "))
odd_number=[]
for i in range(1,max):
    if i%2==1:
        odd_number.append(i)
print(odd_number)'''
#Linked List practice 
class Node:
    def __init__(self,data=None,next=None):
       self.data=data
       self.next=next 
class LinkedList:
     def __init__(self):
           self.head=None
def Insert_at_beginning (self,data):
    node=Node(data,self.head)
    self.head=node
def print(self):

 if __builtins__:
     pass           




 




    





    

