import json
import os
import shutil
import numpy as np

#默认配置文件
CONFIG_FILE="config.json"

def cpfile(src,dst):
    shutil.copyfile(src,dst)
    print("Copied File "+src+" --> "+dst)

with open(CONFIG_FILE, "r", encoding='utf-8') as f:
    jsonfile=json.load(f)
    if(jsonfile["framework"]=='tensorflow'):
        import tensorflow as tf
        print("Use Tensorflow")
        srcdir=jsonfile["paths"]["sourcedir"]
        if(srcdir[-1]!='/'):
            srcdir+='/'
        outputdir=jsonfile["paths"]["outputdir"]
        if(outputdir[-1]!='/'):
            outputdir+='/'
        modeldir=jsonfile["paths"]["modeldir"]
        if(modeldir[-1]!='/'):
            modeldir+='/'
        modelfile=jsonfile["paths"]["modelfile"]
        if(os.path.exists(outputdir+"weights")==False):
            os.makedirs(outputdir+"weights")
        cpfile(srcdir+"activation_function.c",outputdir+"activation_function.c")
        cpfile(srcdir+"activation_function.h",outputdir+"activation_function.h")
        cpfile(srcdir+"basic.c",outputdir+"basic.c")
        cpfile(srcdir+"basic.h",outputdir+"basic.h")
        cpfile(srcdir+"tensor.h",outputdir+"tensor.h")
        USE_CNN=False
        USE_LSTM=False
        wnum=0
        w=[]
        b=[]
        internal_tensor_num=0
        sess=tf.Session()
        saver = tf.train.import_meta_graph(modeldir+modelfile)
        saver.restore(sess,tf.train.latest_checkpoint(modeldir))
        v=tf.trainable_variables()
        for i in jsonfile["nets"]:
            if i["id"]=="CNN":
                USE_CNN=True
                internal_tensor_num+=1
                with open(outputdir+'weights/w'+str(wnum)+'.h','wb') as f:
                    f.write(str('float w'+str(wnum)+"_w[]={\n").encode('utf-8'))
                    arr=np.array(sess.run(v[wnum])).flatten()
                    for x in arr:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
                    w.append(wnum)
                wnum=wnum+1
                with open(outputdir+'weights/b'+str(wnum)+'.h','wb') as f:
                    f.write(str('float b'+str(wnum)+"_w[]={\n").encode('utf-8'))
                    arr=np.array(sess.run(v[wnum])).flatten()
                    for x in arr:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
                    b.append(wnum)
                wnum=wnum+1
            if i["id"]=="LSTM":
                USE_LSTM=True
                internal_tensor_num+=1
                with open(outputdir+'weights/w'+str(wnum)+'.h','wb') as f:
                    arr=np.array(sess.run(v[wnum]))
                    I=[]
                    C=[]
                    F=[]
                    O=[]
                    if len(arr.shape)==2:
                        I=arr[:,0:int(arr.shape[1]/4)]
                        C=arr[:,int(arr.shape[1]/4):int(arr.shape[1]/2)]
                        F=arr[:,int(arr.shape[1]/2):int(arr.shape[1]/2+arr.shape[1]/4)]
                        O=arr[:,int(arr.shape[1]/2+arr.shape[1]/4):int(arr.shape[1])]
                    if len(arr.shape)==3:
                        I=arr[:,:,0:int(arr.shape[2]/4)]
                        C=arr[:,:,int(arr.shape[2]/4):int(arr.shape[2]/2)]
                        F=arr[:,:,int(arr.shape[2]/2):int(arr.shape[2]/2+arr.shape[2]/4)]
                        O=arr[:,:,int(arr.shape[2]/2+arr.shape[2]/4):int(arr.shape[2])]
                    I=np.array(I).flatten()
                    C=np.array(C).flatten()
                    F=np.array(F).flatten()
                    O=np.array(O).flatten()
                    f.write(str('float w'+str(wnum)+"_I[]={\n").encode('utf-8'))
                    for x in I:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    f.write(str('float w'+str(wnum)+"_C[]={\n").encode('utf-8'))
                    for x in C:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    f.write(str('float w'+str(wnum)+"_F[]={\n").encode('utf-8'))
                    for x in F:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    f.write(str('float w'+str(wnum)+"_O[]={\n").encode('utf-8'))
                    for x in O:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    w.append(wnum)
                wnum=wnum+1
                with open(outputdir+'weights/b'+str(wnum)+'.h','wb') as f:
                    arr=np.array(sess.run(v[wnum]))
                    I=[]
                    C=[]
                    F=[]
                    O=[]
                    if len(arr.shape)==2:
                        I=arr[:,0:int(arr.shape[1]/4)]
                        C=arr[:,int(arr.shape[1]/4):int(arr.shape[1]/2)]
                        F=arr[:,int(arr.shape[1]/2):int(arr.shape[1]/2+arr.shape[1]/4)]
                        O=arr[:,int(arr.shape[1]/2+arr.shape[1]/4):int(arr.shape[1])]
                    if len(arr.shape)==3:
                        I=arr[:,:,0:int(arr.shape[2]/4)]
                        C=arr[:,:,int(arr.shape[2]/4):int(arr.shape[2]/2)]
                        F=arr[:,:,int(arr.shape[2]/2):int(arr.shape[2]/2+arr.shape[2]/4)]
                        O=arr[:,:,int(arr.shape[2]/2+arr.shape[2]/4):int(arr.shape[2])]
                    I=np.array(I).flatten()
                    C=np.array(C).flatten()
                    F=np.array(F).flatten()
                    O=np.array(O).flatten()
                    f.write(str('float b'+str(wnum)+"_I[]={\n").encode('utf-8'))
                    for x in I:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    f.write(str('float b'+str(wnum)+"_C[]={\n").encode('utf-8'))
                    for x in C:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    f.write(str('float b'+str(wnum)+"_F[]={\n").encode('utf-8'))
                    for x in F:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    f.write(str('float b'+str(wnum)+"_O[]={\n").encode('utf-8'))
                    for x in O:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};\n").encode('utf-8'))
                    b.append(wnum)
                wnum=wnum+1
            if i["id"]=="Dense":
                internal_tensor_num+=1
                with open(outputdir+'weights/w'+str(wnum)+'.h','wb') as f:
                    f.write(str('float w'+str(wnum)+"_w[]={\n").encode('utf-8'))
                    arr=np.array(sess.run(v[wnum])).flatten()
                    for x in arr:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
                    w.append(wnum)
                wnum=wnum+1
                with open(outputdir+'weights/b'+str(wnum)+'.h','wb') as f:
                    f.write(str('float b'+str(wnum)+"_w[]={\n").encode('utf-8'))
                    arr=np.array(sess.run(v[wnum])).flatten()
                    for x in arr:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
                    b.append(wnum)
                wnum=wnum+1
        if USE_CNN:
            cpfile(srcdir+"CNN.h",outputdir+"CNN.h")
            cpfile(srcdir+"CNN.c",outputdir+"CNN.c")
        if USE_CNN:
            cpfile(srcdir+"LSTM.h",outputdir+"LSTM.h")
            cpfile(srcdir+"LSTM.c",outputdir+"LSTM.c")
        cpfile(srcdir+"net.h",outputdir+"net.h")
        with open(outputdir+'net.c','w',encoding='utf-8') as f:
            f.write('''#include "net.h"'''+'\n')
            if USE_CNN:
                f.write('''#include "CNN.h"'''+'\n')
            if(USE_LSTM):
                f.write('''#include "LSTM.h"'''+'\n')
            for i in w:
                f.write("#include \"weights/w%d.h\""%(i)+'\n')
            for i in b:
                f.write("#include \"weights/b%d.h\""%(i)+'\n')
            f.write('''#include <stdlib.h>'''+'\n\n')
            for i in w:
                f.write('tensor w%d;\n'%(i))
            for i in b:
                f.write('tensor b%d;\n'%(i))
            for i in range(internal_tensor_num):
                f.write('tensor t%d;\n'%(i))
            f.write('\n\n')
            f.write('void global_net_init()\n{\n')
            lastoutsize=jsonfile['inputshape'][0]
            count=0
            for i in jsonfile['nets']:
                if i['id']=='reshape':
                    lastoutsize=i['outshape'][-1]
                if i['id']=='CNN':
                    f.write('w%d.dims[0]=%d;\n'%(count,i['outsize']))
                    f.write('w%d.dims[1]=%d;\n'%(count,lastoutsize))
                    f.write('w%d.dims[2]=%d;\n'%(count,i['coresize'][1]))
                    f.write('w%d.dims[3]=%d;\n'%(count,i['coresize'][0]))
                    f.write('w%d.max_dim=4;\n'%(count))
                    f.write('w%d.data=w%d_w;\n'%(count,count))
                    count=count+1
                    f.write('b%d.dims[0]=%d;\n'%(count,i['outsize']))
                    f.write('b%d.max_dim=1;\n'%(count))
                    f.write('b%d.data=b%d_w;\n'%(count,count))
                    count=count+1
                    lastoutsize=i['outsize']
                if i['id']=='Dense':
                    f.write('w%d.dims[0]=%d;\n'%(count,i['outsize']))
                    f.write('w%d.dims[1]=%d;\n'%(count,lastoutsize))
                    f.write('w%d.max_dim=2;\n'%(count))
                    f.write('w%d.data=w%d_w;\n'%(count,count))
                    count=count+1
                    f.write('b%d.dims[0]=%d;\n'%(count,i['outsize']))
                    f.write('b%d.max_dim=1;\n'%(count))
                    f.write('b%d.data=b%d_w;\n'%(count,count))
                    count=count+1
                    lastoutsize=i['outsize']
                if i['id']=='LSTM':
                    pass
            f.write('}\n\nvoid run_net(tensor* in,tensor* out)\n{\nint i;\nint temp_shape[MAX_DIMENSION];\n')
            laststring='in'
            usedot=False
            tn=0
            count=0
            for idx,i in enumerate(jsonfile['nets']):
                if i['id']=='reshape':
                    frontstr='&' if usedot else ' '
                    for j in range(len(i['outshape'])):
                        f.write('temp_shape[%d]=%d;\n'%(j,i['outshape'][-j-1]))
                    f.write('Reshape(%s,temp_shape,%d);\n'%(frontstr+laststring,len(i['outshape'])))
                    if idx==len(jsonfile['nets'])-1:
                        f.write('out=%s;\n}\n\n'%(frontstr+laststring))
                    lastoutsize=i['outshape'][-1]
                if i['id']=='CNN':
                    dot='.' if usedot else '->'
                    frontstr='&' if usedot else ' '
                    f.write('t%d.data=(float*)malloc(sizeof(float)*(%d*%sdims[1]*%sdims[2]*%sdims[3]));\n'%(tn,i['outsize'],laststring+dot,laststring+dot,laststring+dot))
                    tn=tn+1
                    f.write('Conv2D(%s,&t%d,&w%d);\n'%(frontstr+laststring,tn,count))
                    count+=1
                    if usedot==True:
                        f.write('free(%s.data);\n'%(laststring))
                    f.write('AddVector(&t%d,&b%d,&t%d);\n'%(tn,count,tn))
                    f.write('_activationfunction(_%s,&t%d,&t%d);\n'%(i['activation'],tn,tn))
                    count+=1
                    if(i['pool']=='maxpool2x2'):
                        f.write('maxpool2x2(&t%d,&t%d);\n'%(tn,tn))
                    usedot=True
                    laststring='t%d'%(tn)
                    lastoutsize=i['outsize']
                if i['id']=='Dense':
                    dot='.' if usedot else '->'
                    frontstr='&' if usedot else ' '
                    f.write('t%d.data=(float*)malloc(sizeof(float)*(%sdims[0]*%sdims[1]));\n'%(tn,laststring+dot,laststring+dot))
                    f.write('MatMul(&t%d,&w%d,%s);\n'%(tn,count,frontstr+laststring))
                    count+=1
                    if usedot==True:
                        f.write('free(%s.data);\n'%(laststring))
                    f.write('AddVector(&t%d,&b%d,&t%d);\n'%(tn,count,tn))
                    if idx!=len(jsonfile['nets'])-1:
                        f.write('_activationfunction(_%s,&t%d,&t%d);\n'%(i['activation'],tn,tn))
                    else:
                        f.write('_activationfunction(_%s,&t%d,out);\n'%(i['activation'],tn))
                        f.write('free(&t%d.data);\n}\n\n'%(tn))
                    count+=1
                    tn=tn+1
                    usedot=True
                    laststring='t%d'%(tn)
                    lastoutsize=i['outsize']
            f.write('void global_net_destroy()\n{\n')
            f.write('}\n')
        sess.close()
