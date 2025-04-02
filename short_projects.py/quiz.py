print("Welcome to my computer quiz")
playing=input("do you want to play: ")
if  playing.lower()!="yes":
    quit()
print("okay let's play :)")
score=0
answer=input("what does CPU stand for:").title()
if answer=="Central Processing Unit":
    print("correct!")
    score +=1
else:
    print("incorrect!")
answer=input("what does RAM stand for:").title()
if answer=="Random Access Memory":
    print("correct!")
    score +=1
else:
    print("incorrect!")
answer=input("what does ROM stand for:").title()
if answer=="Read Only Memory":
    print("correct!")
    score +=1
else:
    print("incorrect!")
answer=input("what does HDD stand for:").title()
if answer=="Hard Drive Disk":
    print("correct!")
    score +=1
else:
    print("incorrect!")
print("thank you for playing "+ str(score) +"  rquestions correct ")
