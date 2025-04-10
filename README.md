This repository aims to automate the discovery of algorithms through the use of various machine learning techniques. We start by exploring sparse coding and see how a Neural Architecture Search (NAS) based training framework can recreate algorithms such as the Fast Iterative Shrinkage Thresholding Algorithm (FISTA). 

## **NAS FISTA**

We rediscover FISTA by creating sufficiently large DARTS cells and have the model learn operations matching the FISTA algorithm's operations. The main choices to consider are the proximal operator (shrinkage), the gradient's choice for acceleration (momentum), and the number of layers in our model (iterations of the algorithm). 

## **Preconditioning**

## **Shift Varying Systems**

## **Citation**
```
@misc{Darpa-Disco,
  author = {Sarthak Gupta, Yaoteng Tan, Nebiyou Yismaw, Patrick Yubeaton, Salman Asif, Chinmay Hegde},
  month = april,
  title = {{Darpa-Disco}},
  year = {2025}
```
