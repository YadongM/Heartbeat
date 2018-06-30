# 基于CNN与时域变化的心脏音频识别

------

之前做全国并行应用挑战赛时候的代码

分为并行和串行的代码上

如果有疑问欢迎进行批评改进

```
│  BN_layer.c
│  BN_layer.h
│  cnn.c
│  cnn.cbp
│  cnn.depend
│  cnn.h
│  cnn.layout
│  conv_layer.c
│  conv_layer.h
│  dense_layer.c
│  dense_layer.h
│  list.txt
│  main.c
│  Makefile
│  pool_layer.c
│  pool_layer.h
│  README.md
│  read_npy.c
│  read_npy.h
│  relu_layer.c
│  relu_layer.h
│  softmax.c
│  softmax.h
│  
├─bin
│  └─Debug
│          cnn.exe
│          
├─obj
│  └─Debug
│          BN_layer.o
│          cnn.o
│          conv_layer.o
│          dense_layer.o
│          main.o
│          pool_layer.o
│          read_npy.o
│          relu_layer.o
│          softmax.o
│          softmax_layer.o
│          
├─parallel
│      BN_layer.c
│      BN_layer.h
│      cnn.c
│      cnn.h
│      conv_layer.c
│      conv_layer.h
│      dense_layer.c
│      dense_layer.h
│      main.c
│      Makefile
│      pool_layer.c
│      pool_layer.h
│      read_npy.c
│      read_npy.h
│      relu_layer.c
│      relu_layer.h
│      softmax.c
│      softmax.h
│      
└─serial
    │  BN_layer.c
    │  BN_layer.h
    │  cnn.c
    │  cnn.cbp
    │  cnn.depend
    │  cnn.h
    │  cnn.layout
    │  conv_layer.c
    │  conv_layer.h
    │  dense_layer.c
    │  dense_layer.h
    │  main.c
    │  Makefile
    │  pool_layer.c
    │  pool_layer.h
    │  read_npy.c
    │  read_npy.h
    │  relu_layer.c
    │  relu_layer.h
    │  softmax.c
    │  softmax.h
    │  
    ├─bin
    │  └─Debug
    │          cnn.exe
    │          
    └─obj
        └─Debug
                BN_layer.o
                cnn.o
                conv_layer.o
                dense_layer.o
                main.o
                pool_layer.o
                read_npy.o
                relu_layer.o
                softmax.o
                softmax_layer.o
                
                
```