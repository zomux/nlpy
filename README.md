Natural Language Processing on Python
===


planning

Raphael Shu

Suggested command line usage
===

```
PYTHONPATH="." THEANO_FLAGS='floatX=float32,nvcc.fastmath=True,openmp=True,openmp_elemwise_minsize=1000' \
OMP_NUM_THREADS=8 python ...
```
