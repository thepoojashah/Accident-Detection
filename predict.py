import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alexnet import AlexNet
from caffe_classes_copy import class_names
from socket import *
import threading
import base64
import MySQLdb as db

"""HOST = '192.168.43.163'
PASSWORD = 'animuku'
PORT = 3306
USER = "anirudh"
DB = "rapid"
connection = db.Connection(host=HOST, port = PORT, passwd = PASSWORD, user=USER, db=DB)
dbhandler = connection.cursor()"""

srvr_sock = socket(AF_INET,SOCK_STREAM)
srvr_sock.bind(('',5200))
srvr_sock.listen(5)
print("Server started....")

class_name = ""
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)




def clientTalking(con):
    
    id=con.recv(1024)
    lon=con.recv(1024)
    lat=con.recv(1024)
    img_data=con.recv(100000000)
    with open("./imageToSave.jpg", "w") as fh:
        fh.write(base64.b64decode(img_data))
    
    current_dir = os.getcwd()
    image_dir = os.path.join(current_dir, 'examples')
    img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    imgs = []
    for f in img_files:
        imgs.append(cv2.imread(f))
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.constant(1.0)
    model = AlexNet(x, keep_prob, 3, [])
    score = model.fc8
    softmax = tf.nn.softmax(score)
    saver = tf.train.Saver()

    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './checkpoints/model_epoch10.ckpt')


        for i, image in enumerate(imgs):
            plt.imshow(image[:,:,::-1])
            img = cv2.resize(image.astype(np.float32), (227,227))
            img -= imagenet_mean
            img = np.expand_dims(img, axis=0)
            pred = sess.run(score, feed_dict={x: img})
            predicted_label = pred.argmax(axis=1)
            if predicted_label == 0:
                class_name = 'No Accident'
            elif predicted_label == 1:
                class_name = 'Accident'
            else:
                class_name = 'Fire'
            print class_name
            """if predicted_label == 1 or predicted_label == 2:
                                                    try:
                                                        
                                                        dbhandler.execute("Insert into aid values("+id+",'"+lon+"','"+lat+"',"+predicted_label+",);")
                                                        dbhandler.commit()
                                                    except Exception as e:
                                                        print e"""
            plt.title("Class: " + class_name)
            plt.axis('off')
            plt.show()
            
    con.close()

tpl_clnt, addr = srvr_sock.accept() 
clientTalking(tpl_clnt)
'''thrd = threading.Thread(target=clientTalking,args=(tpl_clnt,))
                    thrd.start()
                    '''    