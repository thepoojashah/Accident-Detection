import os

file = open("val.txt","w")
cnt=0
for path, subdirs, files in os.walk('./test_images/'):
  for name in files:
    if cnt==0:
      cnt=cnt+1
      continue
    else:
      img_path = os.path.join(path,name)
      t1,t2=os.path.split(img_path)
      w,folder,correct_cat=t1.split('/')
      x,end=t2.split('.')
      if end=='jpg':
        if correct_cat == "Accidents":
          label = 1 
        elif correct_cat == "Fire":
          label = 2
        else:
          label = 0
        temp = img_path + " " + str(label) + "\n"
        file.write(temp)
        labels.append(label)
        cnt=cnt+1
print cnt